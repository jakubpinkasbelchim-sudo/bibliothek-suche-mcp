import { AuthorizationError, type AuthRequest, type OAuthHelpers } from "@cloudflare/workers-oauth-provider";

export type Env = {
  OAUTH_PROVIDER: OAuthHelpers;
};

/**
 * Vereinfachter /authorize-Handler für einen rein privat genutzten MCP-Server.
 *
 * Es gibt keinen echten Nutzer-Login und keinen Consent-Screen: Jede
 * Autorisierungsanfrage wird sofort automatisch genehmigt. Das erfüllt nur
 * das OAuth/MCP-Protokoll-Handshake, das Claude beim Verbinden erwartet –
 * es ist KEIN echter Zugriffsschutz. Wer die Worker-URL kennt, kann sich
 * verbinden. Für den persönlichen Gebrauch mit einer geheimen, schwer zu
 * erratenden Worker-URL ist das ein vertretbarer Kompromiss.
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname !== "/authorize") {
      return new Response("Nicht gefunden", { status: 404 });
    }

    let oauthRequest: AuthRequest;
    try {
      oauthRequest = await env.OAUTH_PROVIDER.parseAuthRequest(request);
    } catch (fehler) {
      if (!(fehler instanceof AuthorizationError)) throw fehler;
      if (!fehler.redirectUri) {
        // Unbekannter Client oder ungültige Redirect-URI -> lokal anzeigen,
        // NIEMALS in diesem Fall weiterleiten.
        return new Response(fehler.description, { status: 400 });
      }
      const redirect = new URL(fehler.redirectUri);
      redirect.searchParams.set("error", fehler.code);
      redirect.searchParams.set("error_description", fehler.description);
      if (fehler.state) redirect.searchParams.set("state", fehler.state);
      if (fehler.issuer) redirect.searchParams.set("iss", fehler.issuer);
      return Response.redirect(redirect.toString(), 302);
    }

    // Kein Login, kein Consent-Screen: sofort genehmigen.
    const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
      request: oauthRequest,
      userId: "jakub",
      metadata: { clientName: "Bibliothek-Suche (privat, ohne Login)" },
      scope: oauthRequest.scope,
      props: {},
    });

    return Response.redirect(redirectTo, 302);
  },
};
