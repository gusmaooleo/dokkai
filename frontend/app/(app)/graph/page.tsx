import { Suspense } from "react";

import { GraphScreen } from "@/components/graph/graph-screen";

// `GraphScreen` reads the `?focus=` deep link via `useSearchParams`, which
// requires a Suspense boundary for the production build (see Next's
// "missing Suspense boundary with useSearchParams" error).
export default function GraphPage() {
  return (
    <Suspense fallback={<div className="h-full" />}>
      <GraphScreen />
    </Suspense>
  );
}
