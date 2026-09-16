import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export type Env = {
  AI: Ai;
  CHROMA_API_KEY: string;
  CHROMA_TENANT: string;
  CHROMA_DATABASE: string;
  CHROMA_COLLECTION_ID: string;
};

type ChromaQueryResponse = {
  documents?: (string | null)[][];
  metadatas?: (Record<string, unknown> | null)[][];
  distances?: (number | null)[][];
};

/**
 * MCP-Server, der eine einzige Fähigkeit bereitstellt: semantische Suche
 * über die per Ingest-/Such-Notebook aufgebaute Chroma-Cloud-Collection
 * (Calibre-Bibliothek, Embedding-Modell bge-m3, Distanzmetrik cosine).
 */
export class BibliothekMCP extends McpAgent<Env> {
  server = new McpServer({
    name: "Bibliothek-Suche",
    version: "1.0.0",
  });

  async init() {
    this.server.tool(
      "durchsuche_bibliothek",
      "Durchsucht die Calibre-Bibliothek semantisch nach einer Frage oder einem Thema " +
        "und liefert passende Textstellen mit Titel, Autor und Quellenlink.",
      {
        frage: z.string().describe("Die Suchanfrage in natürlicher Sprache, z. B. 'Wie entsteht Regen?'"),
        anzahl_treffer: z
          .number()
          .int()
          .min(1)
          .max(20)
          .optional()
          .describe("Wie viele Treffer zurückgegeben werden sollen (Standard: 5)"),
      },
      async ({ frage, anzahl_treffer }) => {
        try {
          const vektor = await this.embedde(frage);
          const treffer = await this.frageChroma(vektor, anzahl_treffer ?? 5);
          return {
            content: [{ type: "text", text: this.formatiereTreffer(treffer) }],
          };
        } catch (fehler) {
          return {
            content: [
              {
                type: "text",
                text: `Fehler bei der Suche: ${fehler instanceof Error ? fehler.message : String(fehler)}`,
              },
            ],
          };
        }
      }
    );
  }

  private async embedde(text: string): Promise<number[]> {
    const antwort = await this.env.AI.run("@cf/baai/bge-m3", {
      text: [text],
    });
    const vektor = (antwort as { data?: number[][] })?.data?.[0];
    if (!vektor) {
      throw new Error("Workers AI hat keinen Embedding-Vektor zurückgegeben.");
    }
    return vektor;
  }

  private async frageChroma(vektor: number[], anzahl: number): Promise<ChromaQueryResponse> {
    // Werte trimmen und prüfen, damit Konfigurationsfehler eine klare Meldung
    // liefern statt eines kryptischen Chroma-Fehlers.
    const tenant = (this.env.CHROMA_TENANT ?? "").trim();
    const datenbank = (this.env.CHROMA_DATABASE ?? "").trim();
    const collectionId = (this.env.CHROMA_COLLECTION_ID ?? "").trim();
    const apiKey = (this.env.CHROMA_API_KEY ?? "").trim();

    const fehlend = [
      !tenant && "CHROMA_TENANT",
      !datenbank && "CHROMA_DATABASE",
      !collectionId && "CHROMA_COLLECTION_ID",
      !apiKey && "CHROMA_API_KEY",
    ].filter(Boolean);
    if (fehlend.length > 0) {
      throw new Error(`Secret(s) nicht gesetzt im Worker: ${fehlend.join(", ")}`);
    }

    const uuidV4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    if (!uuidV4.test(collectionId)) {
      throw new Error(
        `CHROMA_COLLECTION_ID ist kein gültiges UUIDv4 (Wert kommt an als: "${collectionId}")`
      );
    }

    const url =
      `https://api.trychroma.com/api/v2/tenants/${tenant}` +
      `/databases/${datenbank}` +
      `/collections/${collectionId}/query`;

    const antwort = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-chroma-token": apiKey,
      },
      body: JSON.stringify({
        query_embeddings: [vektor],
        n_results: anzahl,
        include: ["documents", "metadatas", "distances"],
      }),
    });

    if (!antwort.ok) {
      const fehlertext = await antwort.text();
      throw new Error(`Chroma Cloud antwortete mit ${antwort.status}: ${fehlertext.slice(0, 300)}`);
    }

    return antwort.json();
  }

  private formatiereTreffer(daten: ChromaQueryResponse): string {
    const dokumente = daten.documents?.[0] ?? [];
    const metadaten = daten.metadatas?.[0] ?? [];
    const distanzen = daten.distances?.[0] ?? [];

    if (dokumente.length === 0) {
      return "Keine passenden Textstellen gefunden.";
    }

    return dokumente
      .map((doc, i) => {
        const meta = (metadaten[i] ?? {}) as Record<string, unknown>;
        const distanz = distanzen[i];

        let seite: number | undefined;
        if (typeof meta.location === "string") {
          try {
            seite = JSON.parse(meta.location)?.page;
          } catch {
            /* location nicht parsebar -> ohne Seitenzahl weitermachen */
          }
        }

        const link =
          typeof meta.source_path === "string"
            ? `${meta.source_path}${seite ? `#page=${seite}` : ""}`
            : undefined;

        const kopf = `**${meta.title ?? "Unbekannter Titel"}** (${meta.author ?? "unbekannt"})`;
        const zeilen = [
          kopf,
          typeof distanz === "number" ? `Distanz: ${distanz.toFixed(3)}` : null,
          link ? `Quelle: ${link}` : null,
          doc?.slice(0, 500) ?? "",
        ].filter((zeile): zeile is string => Boolean(zeile));

        return zeilen.join("\n");
      })
      .join("\n\n---\n\n");
  }
}
