import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lean standalone runtime for the Docker image — see frontend/Dockerfile.
  output: "standalone",
};

export default nextConfig;
