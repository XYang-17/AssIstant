async def cli():
    from base.runner import BANNER, anthropic_config
    from uuid import uuid4

    from AssIstant.core.boost.skillManager import SkillManager
    from AssIstant.core.super.director import Director, AgentRequest
    from AssIstant.core.super.memoryManager import MemoryManager, Role
    from AssIstant.common.config import(SKILLS_DIR, MAX_SKILLS_CHARS,REDIS_URL,
                                     CHROMA_HOST, CHROMA_PORT, CHROMA_PERSIST_DIRECTORY)

    print(BANNER)

    skill_manager = SkillManager(
        root_dir=SKILLS_DIR,
        max_prompt_chars=MAX_SKILLS_CHARS
    )
    skill_manager.load()

    config = anthropic_config()
    director = Director(
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        model=config['model'],
        skill_manager=skill_manager
    )
    memory_manager = MemoryManager(
        redis_url=REDIS_URL,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        chroma_path=CHROMA_PERSIST_DIRECTORY,
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        model=config['model']
    )

    user_id, conv_id = "cli_user", str(uuid4())

    while True:
        try:
            msg = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("\nBye!")
            break

        context = await memory_manager.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in context.recent_message[-5:]
        ] if context.recent_message else None
        req = AgentRequest(
            content=msg,
            user_id=user_id,
            conv_id=conv_id,
            context=context.as_prompt(),
            history=history
        )
        resp = await director.run(req)

        await memory_manager.remember(user_id, conv_id, Role.USER, msg)
        await memory_manager.remember(user_id, conv_id, Role.AGENT, resp.response)

        print(f"\n[{resp.agent_type.value.name}]: {resp.response}\n")

    await memory_manager.close()

if __name__ == "__main__":
    import sys

    if "--cli" in sys.argv:
        from asyncio import run
        run(cli())
    else:
        from uvicorn import run
        from AssIstant.common.config import ASSISTANT_HOST, ASSISTANT_PORT, ASSISTANT_ENV
        run(
            "runner.app:app",
            host=ASSISTANT_HOST,
            port=ASSISTANT_PORT,
            reload=("development" == ASSISTANT_ENV)
        )


