import type { Metadata } from "next";
import { Hanken_Grotesk, Inconsolata, Bricolage_Grotesque } from "next/font/google";
import "./globals.css";

import { Providers } from "./providers";

// Variable fonts: no `weight` needed, the full axis range ships in one file.
const hankenGrotesk = Hanken_Grotesk({
  variable: "--font-hanken-grotesk",
  subsets: ["latin"],
  display: "swap",
});

const inconsolata = Inconsolata({
  variable: "--font-inconsolata",
  subsets: ["latin"],
  display: "swap",
});

const bricolageGrotesque = Bricolage_Grotesque({
  variable: "--font-bricolage-grotesque",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "dokkai",
  description: "Graph-aware retrieval for your codebase.",
  icons: {
    icon: [{ url: "/dokkai-logo.svg", type: "image/svg+xml" }],
  },
};

const THEME_INIT_SCRIPT = `(function(){try{var t=localStorage.getItem('dokkai.theme');if(t!=='dark'&&t!=='light'){t='light';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','light');}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${hankenGrotesk.variable} ${inconsolata.variable} ${bricolageGrotesque.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
