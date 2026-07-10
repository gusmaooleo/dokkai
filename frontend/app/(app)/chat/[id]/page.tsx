import { ChatScreen } from "@/components/chat/chat-screen";

export default async function ChatConversationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return <ChatScreen conversationId={id} />;
}
