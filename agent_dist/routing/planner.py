from .models import MultiRouteDecision, ExecutionPlan

class PlannerAgent:
    def plan(self, decision: MultiRouteDecision) -> ExecutionPlan:
        return ExecutionPlan(
            mode=decision.mode, 
            steps=[], 
            active_agents=decision.scope
        )