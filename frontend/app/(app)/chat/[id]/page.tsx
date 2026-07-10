export default async function ChatConversationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return (
    <div className="p-6">
      <h2 className="font-display text-lg font-semibold text-foreground">Chat</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Conversation {id} — coming in a later step.
      </p>
    </div>
  );
}
