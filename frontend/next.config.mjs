/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://127.0.0.1:9876/:path*',
      },
      {
        source: '/api/edge/:path*',
        destination: 'http://127.0.0.1:7777/:path*',
      },
    ];
  },
};

export default nextConfig;
