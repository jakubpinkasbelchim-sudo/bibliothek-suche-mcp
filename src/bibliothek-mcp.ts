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

type Meta = Record<string, unknown>;

type ChromaQueryResponse = {
  documents?: (string | null)[][];
  metadatas?: (Meta | null)[][];
  distances?: (number | null)[][];
};

// Antwort von /get: flache Arrays (im Gegensatz zu /query)
type ChromaGetResponse = {
  ids?: string[];
  documents?: (string | null)[];
  metadatas?: (Meta | null)[];
};

type ChunkRef = { id: string; meta: Meta; sortKey: number; page: number };

/**
 * MCP-Server über die Chroma-Cloud-Collection der Calibre-Bibliothek
 * (Embedding-Modell bge-m3, Distanzmetrik cosine).
 *
 * Tools:
 *   durchsuche_bibliothek  – semantische Suche (Embedding + /query)
 *   finde_buch             – semantische Suche, aber dedupliziert auf Buchebene;
 *                            liefert den exakten source_path für die Tools unten
 *   liste_buecher_von_autor– alle Bücher eines Autors (exakter Autor-String)
 *   zaehle_chunks          – Anzahl der Chunks eines Buchs (+ Beispiel-Metadaten)
 *   lies_buch              – Chunks eines Buchs der Reihe nach, seitenweise
 *   pruefe_konfiguration   – Diagnose der Umgebungsvariablen
 */
export class BibliothekMCP extends McpAgent<Env> {
  server = new McpServer({
    name: "Bibliothek-Suche",
    version: "1.1.0",
  });

  async init() {
    // ------------------------------------------------------------------
    // 1) Semantische Suche (unverändert)
    // ------------------------------------------------------------------
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
      async ({ frage, anzahl_treffer }) =>
        this.sicher(async () => {
          const vektor = await this.embedde(frage);
          const treffer = await this.frageChroma(vektor, anzahl_treffer ?? 5);
          return this.formatiereTreffer(treffer);
        })
    );

    // ------------------------------------------------------------------
    // 2) Buch finden: semantisch suchen, aber nach Buch (source_path)
    //    deduplizieren. Liefert den exakten source_path, den
    //    zaehle_chunks / lies_buch als Parameter brauchen.
    // ------------------------------------------------------------------
    this.server.tool(
      "finde_buch",
      "Findet Bücher in der Bibliothek anhand von Autor, Titel oder Thema und liefert pro Buch " +
        "den exakten source_path, Titel und Autor. Diesen source_path brauchen zaehle_chunks und lies_buch.",
      {
        suchtext: z.string().describe("Autor, Titel oder Thema, z. B. 'W. G. Sebald' oder 'Die Ringe des Saturn'"),
        max_buecher: z
          .number()
          .int()
          .min(1)
          .max(30)
          .optional()
          .describe("Wie viele verschiedene Bücher höchstens zurückgegeben werden (Standard: 10)"),
      },
      async ({ suchtext, max_buecher }) =>
        this.sicher(async () => {
          const vektor = await this.embedde(suchtext);
          // Mehr Treffer holen als Bücher gewünscht, weil viele Chunks aus
          // demselben Buch stammen können.
          const treffer = await this.frageChroma(vektor, 100, ["metadatas", "distances"]);
          const metadaten = treffer.metadatas?.[0] ?? [];
          const distanzen = treffer.distances?.[0] ?? [];

          const buecher = new Map<string, { meta: Meta; distanz: number; anzahl: number }>();
          metadaten.forEach((m, i) => {
            const meta = (m ?? {}) as Meta;
            const pfad = typeof meta.source_path === "string" ? meta.source_path : null;
            if (!pfad) return;
            const d = typeof distanzen[i] === "number" ? (distanzen[i] as number) : 9;
            const vorhanden = buecher.get(pfad);
            if (!vorhanden) buecher.set(pfad, { meta, distanz: d, anzahl: 1 });
            else {
              vorhanden.anzahl += 1;
              vorhanden.distanz = Math.min(vorhanden.distanz, d);
            }
          });

          if (buecher.size === 0) return "Keine Bücher gefunden.";

          const sortiert = [...buecher.entries()]
            .sort((a, b) => a[1].distanz - b[1].distanz)
            .slice(0, max_buecher ?? 10);

          return sortiert
            .map(
              ([pfad, b], i) =>
                `${i + 1}. **${b.meta.title ?? "Unbekannter Titel"}** (${b.meta.author ?? "unbekannt"})\n` +
                `   beste Distanz: ${b.distanz.toFixed(3)}, Treffer-Chunks: ${b.anzahl}\n` +
                `   source_path: ${pfad}`
            )
            .join("\n\n");
        })
    );

    // ------------------------------------------------------------------
    // 3) Alle Bücher eines Autors (exakter Metadaten-Match auf "author")
    // ------------------------------------------------------------------
    this.server.tool(
      "liste_buecher_von_autor",
      "Listet alle Bücher eines Autors anhand des exakten Autor-Strings aus den Metadaten " +
        "(z. B. 'Sebald, W.G.' – den genauen String liefert finde_buch). Keine Teilstring-Suche.",
      {
        autor: z.string().describe("Exakter Autor-String, wie er in den Metadaten steht"),
      },
      async ({ autor }) =>
        this.sicher(async () => {
          const chunks = await this.holeAlleChunks({ author: { $eq: autor } }, ["metadatas"]);
          if (chunks.length === 0) return `Keine Chunks mit author = "${autor}" gefunden.`;

          const buecher = new Map<string, { titel: string; anzahl: number }>();
          for (const c of chunks) {
            const pfad = typeof c.meta.source_path === "string" ? c.meta.source_path : "(ohne source_path)";
            const b = buecher.get(pfad);
            if (b) b.anzahl += 1;
            else buecher.set(pfad, { titel: String(c.meta.title ?? "Unbekannter Titel"), anzahl: 1 });
          }

          return (
            `${buecher.size} Buch/Bücher von "${autor}" (${chunks.length} Chunks gesamt):\n\n` +
            [...buecher.entries()]
              .sort((a, b) => a[1].titel.localeCompare(b[1].titel))
              .map(([pfad, b]) => `- **${b.titel}** – ${b.anzahl} Chunks\n  source_path: ${pfad}`)
              .join("\n")
          );
        })
    );

    // ------------------------------------------------------------------
    // 4) Chunks eines Buchs zählen
    // ------------------------------------------------------------------
    this.server.tool(
      "zaehle_chunks",
      "Zählt die Chunks eines Buchs in der Bibliothek. Erwartet den exakten source_path " +
        "(liefert finde_buch). Zeigt zusätzlich Beispiel-Metadaten zur Kontrolle der Sortierung.",
      {
        source_path: z.string().describe("Exakter source_path des Buchs aus finde_buch"),
      },
      async ({ source_path }) =>
        this.sicher(async () => {
          const chunks = await this.holeAlleChunks({ source_path: { $eq: source_path } }, ["metadatas"]);
          if (chunks.length === 0) return `Keine Chunks für source_path = "${source_path}" gefunden.`;

          const sortiert = this.sortiere(chunks);
          const erster = sortiert[0];
          const letzter = sortiert[sortiert.length - 1];
          const seiten = sortiert.map((c) => c.page).filter((p) => p > 0);

          return (
            `**${erster.meta.title ?? "Unbekannter Titel"}** (${erster.meta.author ?? "unbekannt"})\n` +
            `Chunks: ${chunks.length}\n` +
            (seiten.length > 0 ? `Seitenbereich: ${Math.min(...seiten)}–${Math.max(...seiten)}\n` : "") +
            `\nErster Chunk (nach Sortierung):\n` +
            `  id: ${erster.id}\n` +
            `  sortKey: ${erster.sortKey}\n` +
            `  metadata: ${JSON.stringify(erster.meta)}\n` +
            `\nLetzter Chunk (nach Sortierung):\n` +
            `  id: ${letzter.id}\n` +
            `  sortKey: ${letzter.sortKey}\n` +
            `  metadata: ${JSON.stringify(letzter.meta)}`
          );
        })
    );

    // ------------------------------------------------------------------
    // 5) Buch der Reihe nach lesen (seitenweise)
    // ------------------------------------------------------------------
    this.server.tool(
      "lies_buch",
      "Liefert die Chunks eines Buchs in Lesereihenfolge, seitenweise. Erwartet den exakten " +
        "source_path (liefert finde_buch). Mit 'ab' weiterblättern.",
      {
        source_path: z.string().describe("Exakter source_path des Buchs aus finde_buch"),
        ab: z.number().int().min(0).optional().describe("Index des ersten Chunks (0-basiert, Standard: 0)"),
        anzahl: z
          .number()
          .int()
          .min(1)
          .max(50)
          .optional()
          .describe("Wie viele Chunks zurückgegeben werden (Standard: 20, max. 50)"),
      },
      async ({ source_path, ab, anzahl }) =>
        this.sicher(async () => {
          const start = ab ?? 0;
          const n = anzahl ?? 20;

          // Schritt 1: alle Chunk-IDs + Metadaten des Buchs holen (ohne Texte),
          // sortieren, Ausschnitt bestimmen.
          const alle = this.sortiere(
            await this.holeAlleChunks({ source_path: { $eq: source_path } }, ["metadatas"])
          );
          if (alle.length === 0) return `Keine Chunks für source_path = "${source_path}" gefunden.`;
          if (start >= alle.length) {
            return `'ab' = ${start} liegt außerhalb des Buchs (${alle.length} Chunks, Indizes 0–${alle.length - 1}).`;
          }
          const ausschnitt = alle.slice(start, start + n);

          // Schritt 2: nur für den Ausschnitt die Texte nachladen.
          const texte = await this.holeChunksPerId(ausschnitt.map((c) => c.id));

          const titel = alle[0].meta.title ?? "Unbekannter Titel";
          const autor = alle[0].meta.author ?? "unbekannt";
          const ende = start + ausschnitt.length - 1;
          const weiter =
            ende + 1 < alle.length
              ? `\n\n---\nWeiter mit ab=${ende + 1} (noch ${alle.length - ende - 1} Chunks).`
              : `\n\n---\nEnde des Buchs erreicht.`;

          const koerper = ausschnitt
            .map((c, i) => {
              const seite = c.page > 0 ? `, Seite ${c.page}` : "";
              return `[Chunk ${start + i}${seite}]\n${texte.get(c.id) ?? "(Text nicht gefunden)"}`;
            })
            .join("\n\n");

          return `**${titel}** (${autor}) – Chunks ${start}–${ende} von ${alle.length}\n\n${koerper}${weiter}`;
        })
    );

    // ------------------------------------------------------------------
    // 6) Diagnose (unverändert)
    // ------------------------------------------------------------------
    this.server.tool(
      "pruefe_konfiguration",
      "Prüft, ob die Chroma-Zugangsdaten im Worker ankommen. Gibt nur Variablennamen und " +
        "Längen zurück, keine Werte.",
      {},
      async () => {
        const erwartet = ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "CHROMA_COLLECTION_ID"];
        const env = this.env as unknown as Record<string, unknown>;
        const alleNamen = Object.keys(env).sort();

        const status = erwartet.map((name) => {
          const wert = env[name];
          if (typeof wert !== "string") return `${name}: FEHLT (Typ: ${typeof wert})`;
          const getrimmt = wert.trim();
          const hinweis = getrimmt.length !== wert.length ? " – enthält Leerzeichen/Zeilenumbruch am Rand!" : "";
          return `${name}: vorhanden, Länge ${wert.length}${hinweis}`;
        });

        return {
          content: [
            {
              type: "text",
              text:
                `Erwartete Variablen:\n${status.join("\n")}\n\n` +
                `Alle im Worker sichtbaren env-Schlüssel:\n${alleNamen.join(", ")}`,
            },
          ],
        };
      }
    );
  }

  // ====================================================================
  // Hilfsfunktionen
  // ====================================================================

  /** Einheitliche Fehlerbehandlung für alle Tools. */
  private async sicher(fn: () => Promise<string>) {
    try {
      return { content: [{ type: "text" as const, text: await fn() }] };
    } catch (fehler) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Fehler: ${fehler instanceof Error ? fehler.message : String(fehler)}`,
          },
        ],
      };
    }
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

  /** Liest und validiert die Konfiguration, gibt Basis-URL und Header zurück. */
  private chromaKonfig(): { basisUrl: string; headers: Record<string, string> } {
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

    return {
      basisUrl:
        `https://api.trychroma.com/api/v2/tenants/${tenant}` +
        `/databases/${datenbank}` +
        `/collections/${collectionId}`,
      headers: { "Content-Type": "application/json", "x-chroma-token": apiKey },
    };
  }

  /** Generischer POST gegen einen Collection-Endpunkt (/query oder /get). */
  private async chromaPost<T>(endpunkt: "query" | "get", body: unknown): Promise<T> {
    const { basisUrl, headers } = this.chromaKonfig();
    const antwort = await fetch(`${basisUrl}/${endpunkt}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!antwort.ok) {
      const fehlertext = await antwort.text();
      throw new Error(`Chroma Cloud antwortete mit ${antwort.status}: ${fehlertext.slice(0, 300)}`);
    }
    return antwort.json() as Promise<T>;
  }

  private async frageChroma(
    vektor: number[],
    anzahl: number,
    include: string[] = ["documents", "metadatas", "distances"]
  ): Promise<ChromaQueryResponse> {
    return this.chromaPost<ChromaQueryResponse>("query", {
      query_embeddings: [vektor],
      n_results: anzahl,
      include,
    });
  }

  /**
   * Holt ALLE Chunks, die einem Metadaten-Filter entsprechen, per Pagination
   * über /get. Texte werden nur geladen, wenn "documents" in include steht.
   */
  private async holeAlleChunks(where: Record<string, unknown>, include: string[]): Promise<ChunkRef[]> {
    const seitenGroesse = 250;
    const maxChunks = 20000; // Sicherheitsgrenze gegen Endlosschleifen / zu breite Filter
    const ergebnis: ChunkRef[] = [];
    let offset = 0;

    while (offset < maxChunks) {
      const seite = await this.chromaPost<ChromaGetResponse>("get", {
        where,
        limit: seitenGroesse,
        offset,
        include,
      });
      const ids = seite.ids ?? [];
      ids.forEach((id, i) => {
        const meta = (seite.metadatas?.[i] ?? {}) as Meta;
        ergebnis.push({ id, meta, sortKey: this.sortKey(id, meta), page: this.seite(meta) });
      });
      if (ids.length < seitenGroesse) break;
      offset += seitenGroesse;
    }
    return ergebnis;
  }

  /** Lädt die Texte zu konkreten Chunk-IDs. */
  private async holeChunksPerId(ids: string[]): Promise<Map<string, string>> {
    const texte = new Map<string, string>();
    if (ids.length === 0) return texte;
    const antwort = await this.chromaPost<ChromaGetResponse>("get", {
      ids,
      include: ["documents"],
    });
    (antwort.ids ?? []).forEach((id, i) => {
      texte.set(id, antwort.documents?.[i] ?? "");
    });
    return texte;
  }

  /** Seitenzahl aus dem location-JSON, 0 wenn nicht vorhanden. */
  private seite(meta: Meta): number {
    const loc = this.parseLocation(meta);
    const p = loc?.page;
    return typeof p === "number" ? p : typeof p === "string" && /^\d+$/.test(p) ? Number(p) : 0;
  }

  private parseLocation(meta: Meta): Meta | null {
    if (typeof meta.location === "string") {
      try {
        const v = JSON.parse(meta.location);
        return v && typeof v === "object" ? (v as Meta) : null;
      } catch {
        return null;
      }
    }
    if (meta.location && typeof meta.location === "object") return meta.location as Meta;
    return null;
  }

  /**
   * Bestimmt einen Sortierschlüssel für die Lesereihenfolge. Da beim Ingest
   * kein ausdrücklicher Chunk-Index bekannt ist, werden mehrere Kandidaten
   * geprüft:
   *   1. numerische Felder in den Metadaten (chunk_index, chunk, index, chunk_id, position, seq)
   *   2. dieselben Felder innerhalb von location
   *   3. Zeichen-/Byte-Offsets in location (start, offset, char_start)
   *   4. eine Zahl am Ende der Chunk-ID (z. B. "...-123" oder "...::123")
   *   5. sonst Seitenzahl * 1_000_000 (grobe Ordnung)
   * zaehle_chunks zeigt id und sortKey des ersten/letzten Chunks, damit
   * sich prüfen lässt, ob die Reihenfolge stimmt.
   */
  private sortKey(id: string, meta: Meta): number {
    const kandidaten = ["chunk_index", "chunk_idx", "chunk", "index", "chunk_id", "position", "seq", "chunk_no"];
    const zahl = (v: unknown): number | null => {
      if (typeof v === "number" && Number.isFinite(v)) return v;
      if (typeof v === "string" && /^\d+$/.test(v)) return Number(v);
      return null;
    };

    for (const k of kandidaten) {
      const z = zahl(meta[k]);
      if (z !== null) return z;
    }
    const loc = this.parseLocation(meta);
    if (loc) {
      for (const k of [...kandidaten, "start", "offset", "char_start", "start_char"]) {
        const z = zahl(loc[k]);
        if (z !== null) return z;
      }
    }
    const m = id.match(/(\d+)\s*$/);
    if (m) return Number(m[1]);

    return this.seite(meta) * 1_000_000;
  }

  private sortiere(chunks: ChunkRef[]): ChunkRef[] {
    return [...chunks].sort(
      (a, b) => a.sortKey - b.sortKey || a.page - b.page || a.id.localeCompare(b.id, undefined, { numeric: true })
    );
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
        const meta = (metadaten[i] ?? {}) as Meta;
        const distanz = distanzen[i];
        const seite = this.seite(meta);

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
