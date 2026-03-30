import re

_DEVELOPER_ROLE_RE = re.compile(r"""==\s*['"]developer['"]""")


def map_developer_role(
    messages: list[dict[str, str]], chat_template: str | None
) -> list[dict[str, str]]:
    """Map ``developer`` role to ``system`` when the chat template doesn't support it.

    Modern coding agents (Claude Code, Cursor, Aider, etc.) send messages with
    the ``developer`` role.  Most Jinja chat templates only know about
    ``system`` and will raise a ``TemplateError`` for unknown roles.  This
    helper rewrites ``developer`` → ``system`` unless the template explicitly
    handles the role (detected by an ``== 'developer'`` / ``== "developer"``
    comparison in the template source).
    """
    has_developer = any(m.get("role") == "developer" for m in messages)
    if not has_developer:
        return messages
    if chat_template and _DEVELOPER_ROLE_RE.search(chat_template):
        return messages
    return [
        {**m, "role": "system"} if m.get("role") == "developer" else m for m in messages
    ]
