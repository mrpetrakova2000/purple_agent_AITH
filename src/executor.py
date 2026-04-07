from a2a.server.executors import AgentExecutor
from agent import Agent

class Executor(AgentExecutor):
    """A2A executor that delegates to your Agent implementation."""
    
    def __init__(self):
        self.agent = Agent()
    
    async def execute(self, message, context):
        """Execute the agent logic for an incoming message."""
        updater = context.updater
        await self.agent.run(message, updater)