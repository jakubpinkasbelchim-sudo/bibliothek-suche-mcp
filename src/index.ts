import { OAuthProvider, type OAuthHelpers } from "@cloudflare/workers-oauth-provider";
import authHandler from "./auth-handler";
import { BibliothekMCP } from "./bibliothek-mcp";

// Die Durable-Object-Klasse muss aus dem Haupt-Modul exportiert werden,
// damit Wrangler sie an den in wrangler.jsonc definierten Namen binden kann.
export { BibliothekMCP };

export interface Env {
  AI: Ai;
  OAUTH_KV: KVNamespace;
  OAUTH_PROVIDER: OAuthHelpers;
  CHROMA_API_KEY: string;
  CHROMA_TENANT: string;
  CHROMA_DATABASE: string;
  CHROMA_COLLECTION_ID: string;
}

// Diese URL bitte anpassen, falls sich der Worker-Name/Account-Subdomain
// jemals ändert.
const WORKER_URL = "https://bibliothek-suche-mcp.jakub-pinkas-belchim.workers.dev";

export default new OAuthProvider<Env>({
  // Der eigentliche MCP-Endpunkt: nur mit gültigem (automatisch erteiltem)
  // Token erreichbar.
  apiRoute: "/mcp",
  apiHandler: BibliothekMCP.serve("/mcp"),

  // Alles andere (Autorisierung, Token, Registrierung) läuft über unseren
  // eigenen, sehr einfachen Auto-Approve-Handler.
  defaultHandler: authHandler,

  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/oauth/token",
  // Dynamic Client Registration: Claude registriert sich beim ersten
  // Verbindungsversuch automatisch selbst über diesen Endpunkt.
  clientRegistrationEndpoint: "/oauth/register",

  resourceMetadata: {
    resource: `${WORKER_URL}/mcp`,
    authorization_servers: [WORKER_URL],
    resource_name: "Bibliothek-Suche",
  },
});
