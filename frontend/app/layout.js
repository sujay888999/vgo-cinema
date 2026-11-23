import './globals.css';
import Providers from './providers';

export const metadata = {
  title: 'VGO Cinema - AI Video Generation',
  description: 'Turn your vision into cinematic masterpieces',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}

export const viewport = {
  width: "device-width",
  initialScale: 1,
};
