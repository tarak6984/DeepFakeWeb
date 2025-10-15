import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Add rewrites to proxy API calls to local deepfake server
  async rewrites() {
    return [
      {
        source: '/api/deepfake/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Warning: This allows production builds to successfully complete even if
    // your project has type errors.
    ignoreBuildErrors: false,
    // Speed up development by running type checking in separate process
    tsconfigPath: './tsconfig.json',
  },
  // Performance optimizations for faster development
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },
  // Optimize bundle analysis
  experimental: {
    optimizePackageImports: ['lucide-react', '@radix-ui/react-icons', 'recharts', 'framer-motion'],
  },
  // Speed up builds
  typedRoutes: false,
  serverExternalPackages: ['jspdf', 'html2canvas'],
  // Turbopack configuration (replaces the deprecated experimental.turbo)
  turbopack: {
    rules: {
      '*.svg': {
        loaders: ['@svgr/webpack'],
        as: '*.js',
      },
    },
  },
  // Webpack optimizations (fallback for when not using Turbopack)
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      // Speed up development builds
      config.cache = {
        type: 'filesystem',
        buildDependencies: {
          config: [__filename],
        },
      };
      // Reduce bundle size in development
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          default: false,
          vendors: false,
          framework: {
            chunks: 'all',
            name: 'framework',
            test: /(?<!node_modules.*)[\\/]node_modules[\\/](react|react-dom|scheduler|prop-types|use-subscription)[\\/]/,
            priority: 40,
            enforce: true,
          },
          lib: {
            test: /[\\/]node_modules[\\/]/,
            name: 'lib',
            priority: 30,
            minChunks: 1,
            reuseExistingChunk: true,
          },
        },
      };
    }
    return config;
  },
};

export default nextConfig;
