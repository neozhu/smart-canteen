import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Canteen - 智能食堂",
  description: "AI-powered vision checkout system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">{children}</body>
    </html>
  );
}
