import Link from "next/link";

/**
 * Placeholder routine run detail page — the step timeline + Findings/Summary
 * tabs (decision 9e) land in a later step (C4). This just confirms
 * navigation from the launch card / run history works.
 */
export default async function RoutineRunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return (
    <div className="mx-auto max-w-[860px] px-3.5 py-[18px] sm:px-6.5 sm:py-[30px]">
      <Link href="/routines" className="text-[13px] text-accent hover:underline">
        ← Back to Routines
      </Link>
      <p className="mt-4 font-mono text-[13px] text-muted-foreground">
        Routine run <span className="text-foreground">{id}</span> — detail view coming soon.
      </p>
    </div>
  );
}
