from Agent.knowledge_agent import KnowledgeAgent
def get_student_report(conversations):
    report = {}
    report["conversations_cnt"] = len(conversations)
    messages_cnt = 0
    all_messages = ""
    for conversation in conversations:
        for msg in conversation["messages"]:
            all_messages += f"{msg['role']}: {msg['content']}\n"
            messages_cnt += 1

    report["messages_cnt"] = messages_cnt

    all_messages = all_messages.strip()

    knowledge_agent = KnowledgeAgent()
    result = knowledge_agent.analyze(all_messages)
    report["knowledges"] = result['explicit']

    return report