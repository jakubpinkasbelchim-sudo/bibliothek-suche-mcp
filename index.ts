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
 *
 * Ablauf pro Anfrage:
 *   1. Suchtext mit demselben Modell einbetten, das beim Indexieren
 *      verwendet wurde (@cf/baai/bge-m3 über Workers AI) -> sonst
 *      passt die Vektordimension nicht und die Treffer sind wertlos.
 *   2. Vektor gegen die Chroma-Cloud-Collection abfragen (REST API v2).
 *   3. Treffer inkl. Metadaten (Titel, Autor, Drive-Link mit Seitenzahl)
 *      als Text formatieren.
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

  /** Bettet den Suchtext mit demselben Modell ein, das beim Indexieren benutzt wurde. */
  private async embedde(text: string): Promise<number[]> {
    // Hinweis: @cf/baai/bge-m3 ist dasselbe Modell wie das lokal in Colab
    // verwendete BAAI/bge-m3 -> gleiche Dimension, kein Mismatch mit dem Index.
    const antwort = await this.env.AI.run("@cf/baai/bge-m3", {
      text: [text],
    });
    const vektor = (antwort as { data?: number[][] })?.data?.[0];
    if (!vektor) {
      throw new Error("Workers AI hat keinen Embedding-Vektor zurückgegeben.");
    }
    return vektor;
  }

  /** Fragt die Chroma-Cloud-Collection über die REST API v2 ab. */
  private async frageChroma(vektor: number[], anzahl: number): Promise<ChromaQueryResponse> {
    const url =
      `https://api.trychroma.com/api/v2/tenants/${this.env.CHROMA_TENANT}` +
      `/databases/${this.env.CHROMA_DATABASE}` +
      `/collections/${this.env.CHROMA_COLLECTION_ID}/query`;

    const antwort = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-chroma-token": this.env.CHROMA_API_KEY,
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

  /** Formatiert die Chroma-Antwort zu lesbarem Text mit Quellenangabe. */
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

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);
    if (url.pathname === "/mcp") {
      return BibliothekMCP.serveSSE("/mcp").fetch(request, env, ctx);
    }
    return new Response("Nicht gefunden. MCP-Endpunkt liegt unter /mcp", { status: 404 });
  },
};
