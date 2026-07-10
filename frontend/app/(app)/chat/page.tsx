import { ChatScreen } from "@/components/chat/chat-screen";

export default async function ChatPage({
  searchParams,
}: {
  searchParams: Promise<{ prefill?: string }>;
}) {
  const { prefill } = await searchParams;
  return <ChatScreen initialComposerText={prefill} />;
}
