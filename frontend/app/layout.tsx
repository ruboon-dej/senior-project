import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ThaiSL Translator",
  description: "Thai <-> Thai Sign Language translation mockup",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <header className="border-b bg-white">
          <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-semibold">
              TS
            </div>
            <span className="font-semibold text-lg">ThaiSL Translator</span>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="text-center text-xs text-gray-400 py-4">
          Mockup build &middot; ThSL Senior Project
        </footer>
      </body>
    </html>
  );
}
