from datetime import datetime

class Student:
    def __init__(self, student_id, name, classes=None):
        self.student_id = student_id
        self.name = name
        self.classes = classes or []

    def add_class(self, class_id):
        if class_id not in self.classes:
            self.classes.append(class_id)

    def remove_class(self, class_id):
        if class_id in self.classes:
            self.classes.remove(class_id)





class Conversation:
    def __init__(self, conversation_id, student_id, title, messages, knowledges, created_at, updated_at, summary):
        self.conversation_id = conversation_id
        self.student_id = student_id
        self.title = title
        self.messages = messages
        self.knowledges = knowledges
        self.created_at = created_at
        self.updated_at = updated_at
        self.summary = summary

    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "student_id": self.student_id,
            "title": self.title,
            "messages": self.messages,
            "knowledges": self.knowledges,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            conversation_id=data["conversation_id"],
            student_id=data["student_id"],
            title=data["title"],
            messages=data["messages"],
            knowledges=data["knowledges"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            summary=data["summary"]
        )