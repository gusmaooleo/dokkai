import { RunDetail } from "@/components/routines/run-detail";

export default async function RoutineRunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  return <RunDetail runId={id} />;
}
