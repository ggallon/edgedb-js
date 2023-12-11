import type { Client } from "edgedb";
import {
  Auth,
  type BuiltinOAuthProviderNames,
  type TokenData,
  type emailPasswordProviderName,
} from "@edgedb/auth-core";

import {
  type RemixAuthOptions,
  RemixClientAuth,
  RemixAuthSession,
} from "./client";
import { Octokit } from "@octokit/core";

export type { RemixAuthOptions } from "./client";

export type BuiltinProviderNames =
  | BuiltinOAuthProviderNames
  | typeof emailPasswordProviderName;

export function createServerAuth(client: Client, options: RemixAuthOptions) {
  return new RemixServerAuth(client, options);
}

type ParamsOrError<Result extends object, ErrorDetails extends object = {}> =
  | ({ error: null } & { [Key in keyof ErrorDetails]?: undefined } & Result)
  | ({ error: Error } & ErrorDetails & { [Key in keyof Result]?: undefined });

export interface CreateAuthRouteHandlers {
  onBuiltinUICallback(
    params: ParamsOrError<
      (
        | {
            tokenData: TokenData;
            provider: BuiltinProviderNames;
            headers: Headers;
          }
        | { tokenData: null; provider: null; headers: null }
      ) & { isSignUp: boolean }
    >
  ): Promise<Response>;
  onEmailVerify(
    params: ParamsOrError<
      { tokenData: TokenData; headers: Headers },
      { verificationToken?: string }
    >
  ): Promise<Response>;
  onSignout(params: { headers: Headers }): void;
}

export class RemixServerAuth extends RemixClientAuth {
  private readonly client: Client;
  private readonly core: Promise<Auth>;

  constructor(client: Client, options: RemixAuthOptions) {
    super(options);

    this.client = client;
    this.core = Auth.create(client);
  }

  isPasswordResetTokenValid(resetToken: string) {
    return Auth.checkPasswordResetTokenValid(resetToken);
  }

  getSession(request: Request) {
    return new RemixAuthSession(
      this.client,
      parseCookies(request)[this.options.authCookieName]
    );
  }

  async getProvidersInfo() {
    return (await this.core).getProvidersInfo();
  }

  createAuthRouteHandlers({
    // onOAuthCallback,
    onBuiltinUICallback,
    onEmailVerify,
    onSignout,
  }: Partial<CreateAuthRouteHandlers>) {
    return {
      loader: async ({
        request,
        params,
      }: {
        request: Request;
        params: { [key: string]: string | undefined };
      }) => {
        const path = params["*"];
        if (!path) {
          throw new Error(`route handlers file should end with '.$.ts'`);
        }
        const searchParams = new URL(request.url).searchParams;
        switch (path) {
          case "oauth": {
            return null; // todo
          }
          case "oauth/callback": {
            return null; // todo
          }
          case "builtin/callback": {
            if (!onBuiltinUICallback) {
              throw new Error(
                `'onBuiltinUICallback' auth route handler not configured`
              );
            }
            const error = searchParams.get("error");
            if (error) {
              const desc = searchParams.get("error_description");
              return onBuiltinUICallback({
                error: new Error(error + (desc ? `: ${desc}` : "")),
              });
            }
            const code = searchParams.get("code");
            const verificationEmailSentAt = searchParams.get(
              "verification_email_sent_at"
            );

            if (!code) {
              if (verificationEmailSentAt) {
                return onBuiltinUICallback({
                  error: null,
                  tokenData: null,
                  provider: null,
                  headers: null,
                  isSignUp: true,
                });
              }
              return onBuiltinUICallback({
                error: new Error("no pkce code in response"),
              });
            }
            const verifier =
              parseCookies(request)[this.options.pkceVerifierCookieName];

            if (!verifier) {
              return onBuiltinUICallback({
                error: new Error("no pkce verifier cookie found"),
              });
            }
            const isSignUp = searchParams.get("isSignUp") === "true";
            let tokenData: TokenData;
            try {
              tokenData = await (await this.core).getToken(code, verifier);
            } catch (err) {
              return onBuiltinUICallback({
                error: err instanceof Error ? err : new Error(String(err)),
              });
            }

            const headers = new Headers();
            headers.append(
              "Set-Cookie",
              `${this.options.authCookieName}=${tokenData.auth_token}; HttpOnly; SameSite=Strict; Path=/`
            );
            headers.append(
              "Set-Cookie",
              `${this.options.pkceVerifierCookieName}=`
            );

            return onBuiltinUICallback({
              error: null,
              tokenData,
              provider: searchParams.get("provider") as BuiltinProviderNames,
              isSignUp,
              headers,
            });
          }
          case "builtin/signin":
          case "builtin/signup": {
            const pkceSession = await this.core.then((core) =>
              core.createPKCESession()
            );
            return new Response(null, {
              status: 302,
              headers: {
                "Set-Cookie": `${this.options.pkceVerifierCookieName}=${pkceSession.verifier}; HttpOnly`,
                Location:
                  path.split("/").pop() === "signup"
                    ? pkceSession.getHostedUISignupUrl()
                    : pkceSession.getHostedUISigninUrl(),
              },
            });
          }
          case "emailpassword/verify": {
            if (!onEmailVerify) {
              throw new Error(
                `'onEmailVerify' auth route handler not configured`
              );
            }
            const verificationToken = searchParams.get("verification_token");
            const verifier =
              parseCookies(request)[this.options.pkceVerifierCookieName];
            if (!verificationToken) {
              return onEmailVerify({
                error: new Error("no verification_token in response"),
              });
            }
            if (!verifier) {
              return onEmailVerify({
                error: new Error("no pkce verifier cookie found"),
                verificationToken,
              });
            }
            let tokenData: TokenData;
            try {
              tokenData = await (
                await this.core
              ).verifyEmailPasswordSignup(verificationToken, verifier);
            } catch (err) {
              return onEmailVerify({
                error: err instanceof Error ? err : new Error(String(err)),
                verificationToken,
              });
            }

            const headers = new Headers();
            headers.append(
              "Set-Cookie",
              `${this.options.authCookieName}=${tokenData.auth_token}; HttpOnly; SameSite=strict; Path=/`
            );

            return onEmailVerify({
              error: null,
              tokenData,
              headers,
            });
          }

          case "signout": {
            if (!onSignout) {
              throw new Error(`'onSignout' auth route handler not configured`);
            }

            const headers = new Headers({
              ...request.headers,
              "Set-Cookie": `${
                this.options.authCookieName
              }=; HttpOnly; SameSite=strict; Expires=${Date.now()}; Path=/`,
            });

            return onSignout({ headers });
          }

          default:
            throw new Error("Unknown auth route");
        }
      },
    };
  }

  async signupWithEmailPassword(request: Request) {
    const formData = await request.formData();

    const [email, password] = _extractParams(
      formData,
      ["email", "password"],
      "email or password missing"
    );

    const result = await (
      await this.core
    ).signupWithEmailPassword(
      email,
      password,
      `${this._authRoute}/emailpassword/verify`
    );

    const headers = new Headers();
    headers.append(
      "Set-Cookie",
      `${this.options.pkceVerifierCookieName}=${result.verifier}; HttpOnly; SameSite=strict`
    );

    if (result.status === "complete") {
      const tokenData = result.tokenData;

      headers.append(
        "Set-Cookie",
        `${this.options.authCookieName}=${tokenData.auth_token}; HttpOnly; SameSite=strict`
      );
      return { tokenData, headers };
    }

    return { tokenData: null, headers };
  }

  async emailPasswordResendVerificationEmail(
    data: FormData | { verification_token: string }
  ) {
    const [verificationToken] = _extractParams(
      data,
      ["verification_token"],
      "verification_token missing"
    );

    await (await this.core).resendVerificationEmail(verificationToken);
  }

  async emailPasswordSignIn(request: Request) {
    const formData = await request.formData();

    const [email, password] = _extractParams(
      formData,
      ["email", "password"],
      "email or password missing"
    );

    const tokenData = await (
      await this.core
    ).signinWithEmailPassword(email, password);

    const headers = new Headers({
      "Set-Cookie": `${this.options.authCookieName}=${tokenData.auth_token}; HttpOnly; SameSite=strict`,
    });

    return { tokenData, headers };
  }

  async emailPasswordSendPasswordResetEmail(request: Request) {
    const data = await request.formData();

    if (!this.options.passwordResetPath) {
      throw new Error(`'passwordResetPath' option not configured`);
    }
    const [email] = _extractParams(data, ["email"], "email missing");

    const { verifier } = await (
      await this.core
    ).sendPasswordResetEmail(
      email,
      new URL(this.options.passwordResetPath, this.options.baseUrl).toString()
    );

    const headers = new Headers({
      "Set-Cookie": `${this.options.pkceVerifierCookieName}=${verifier}; HttpOnly; SameSite=strict`,
    });

    return { headers };
  }

  async emailPasswordResetPassword(request: Request) {
    const verifier = parseCookies(request)[this.options.pkceVerifierCookieName];

    if (!verifier) {
      throw new Error("no pkce verifier cookie found");
    }

    let resetToken = new URL(request.url).searchParams.get("reset_token");
    if (Array.isArray(resetToken)) {
      resetToken = resetToken[0];
    }

    if (!resetToken) {
      throw new Error("reset token not found");
    }

    const [password] = _extractParams(
      await request.formData(),
      ["password"],
      "password missing"
    );

    const tokenData = await (
      await this.core
    ).resetPasswordWithResetToken(resetToken, verifier, password);

    const headers = new Headers();

    headers.append(
      "Set-Cookie",
      `${this.options.authCookieName}=${tokenData.auth_token}; HttpOnly; SameSite=Lax`
    );
    headers.append("Set-Cookie", `${this.options.pkceVerifierCookieName}=`);

    return { tokenData, headers };
  }

  async signout(request: Request) {
    return {
      headers: new Headers({
        ...request.headers,
        "Set-Cookie": `${
          this.options.authCookieName
        }=; HttpOnly; SameSite=strict; expires=${Date.now()}`,
      }),
    };
  }

  async createUser(tokenData: TokenData, provider?: BuiltinProviderNames) {
    let username: string | null = null;
    if (tokenData.provider_token && provider === "builtin::oauth_github") {
      const { data } = await new Octokit({
        auth: tokenData.provider_token,
      }).request("get /user");

      username = data.login;
    }
    await this.client.query(
      `
    with identity := (select ext::auth::Identity filter .id = <uuid>$identity_id),
    insert User {
      identity := identity,
      name := <optional str>$username ?? (
        select ext::auth::EmailFactor filter .identity = identity
      ).email
    } unless conflict on .identity`,
      {
        identity_id: tokenData.identity_id,
        username: username,
      }
    );
  }
}

function parseCookies(request: Request) {
  const cookies = request.headers.get("Cookie");

  return cookies
    ? cookies.split(";").reduce((cookies, cookie) => {
        const [name, val] = cookie.split("=");
        cookies[name.trim()] = val.trim();
        return cookies;
      }, {} as { [key: string]: string | undefined })
    : {};
}

function _extractParams(
  data: FormData | Record<string, unknown>,
  paramNames: string[],
  errMessage: string
) {
  const params: string[] = [];
  if (data instanceof FormData) {
    for (const paramName of paramNames) {
      const param = data.get(paramName)?.toString();
      if (!param) {
        throw new Error(errMessage);
      }
      params.push(param);
    }
  } else {
    if (typeof data !== "object") {
      throw new Error("expected json object");
    }
    for (const paramName of paramNames) {
      const param = data[paramName];
      if (!param) {
        throw new Error(errMessage);
      }
      if (typeof param !== "string") {
        throw new Error(`expected '${paramName}' to be a string`);
      }
      params.push(param);
    }
  }

  return params;
}