import asyncio
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Optional, Dict, Any, List
from statistics import mean, stdev
from json import load
from prometheus_client import metrics
from prometheus_client import start_http_server
from httpx import AsyncClient


from base.monitor import *
from AssIstant.core.super.director import Director, ToolManager
from AssIstant.common.config import ALERT_PATH
from AssIstant.common.logger import FormatLog


with open(ALERT_PATH, "r", encoding="utf-8") as f:
    _alert_info = load(f)
    _threshold = _alert_info.get("threshold") or {}
    _prometheus = _alert_info.get("prometheus") or {}


class AnomalyDetector:
    def __init__(self, window: int = 60, sensitivity: float = 2.5):
        self._window = window
        self._sensitivity = sensitivity

        self._history = defaultdict(lambda : deque(maxlen=self._window))

    def record(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        buf = self._history[metric]
        buf.append(value)

        if len(buf) * 2 < self._window:
            return None

        m = mean(buf)
        v = stdev(buf) if len(buf) > 1 else 0.0
        if 0 == v: return None

        z = abs(value - m) / v
        if z > self._sensitivity:
            return {
                "metric": metric,
                "value": value,
                "mean": m,
                "z_score": round(z, 2),
                "severity": "high" if z > self._sensitivity * 1.5 else "medium"
            }
        return None


class Monitor(FormatLog):
    def __init__(
        self,
        director: Director,
        tool_manager: ToolManager,
        interval: float = 10.0,
        webhook_url: Optional[str] = None,
        prometheus_port: Optional[int] = None,
        name: str = "monitor"
    ):
        FormatLog.__init__(self, name)
        self._direct = director
        self._tool_manager = tool_manager
        self._interval = interval
        self._webhook_url = webhook_url

        self._detector = AnomalyDetector()

        self._alerts: List[Alert] = list()
        self._suggestions: List[Suggestion] = list()
        self._active = False
        self._task: Optional[asyncio.Task] = None

        self._prometheus: Dict[str, Any] = dict()
        if prometheus_port:
            self._prometheus_setup(prometheus_port)

    async def start(self):
        if self._active: return
        self._active = True
        self._task = asyncio.create_task(self._loop())
        self.info(f"successfully started (interval={self._interval})")

    async def stop(self):
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                ...

    def summary(self) -> Dict[str, Any]:
        return {
            "agent_states": self._direct.get_states(),
            "tool_states": self._tool_manager.get_states(),
            "active_alerts": [asdict(a) for a in self._alerts if not a.resolved][-10:],
            "suggestions": [
                {"title": s.title, "action": s.action, "priority": s.priority}
                for s in sorted(self._suggestions, key=lambda x: -x.priority)[:5]
            ]
        }

    def _prometheus_setup(self, port: int):
        self._prometheus.clear()
        for k, v in _prometheus.items():
            try:
                self._prometheus[k] = getattr(metrics, v["metric"])(**v["params"])
            except Exception:
                ...
        start_http_server(port)
        self.info(f"successfully started prometheus (port={port})")

    async def _loop(self):
        while self._active:
            try:
                await self._collect()
            except Exception as e:
                self.exception(f"exception occurred: {e}")
            await asyncio.sleep(self._interval)

    async def _collect(self):
        agent_state = self._direct.get_states()
        tool_state = self._tool_manager.get_states()
        routing_penalties: Dict[str, float] = {}

        items = ("success_rate", "average_latency")
        for agent, v in agent_state.items():
            for item in items:
                anomaly = self._detector.record(f"agent_{item}:{agent}", v[item])
                if anomaly: self.warning(f"anomaly [{agent}] agent_{item}={v[item]:.3f} z={anomaly['z_score']}")

            for item in items:
                self._check_threshold(f"agent_{item}", v[item], agent)

            if f"agent_success_rate" in self._prometheus:
                self._prometheus["agent_success_rate"].labels(agent=agent).set(v['success_rate'])
                self._prometheus["agent_average_latency"].labels(agent=agent).observe(v['average_latency']/1000)

            routing_penalties[agent] = self._routing_penalty(v["success_rate"], v["average_latency"])

        for tool, v in tool_state.items():
            for item in items:
                self._check_threshold(f"tool_{item}", v[item], tool)

            if "tool_success_rate" in self._prometheus:
                self._prometheus["agent_success_rate"].labels(tool=tool).set(v['success_rate'])
                self._prometheus["agent_average_latency"].labels(tool=tool).observe(v['average_latency']/1000)

            if (cf := v["consecutive_fails"]) >= 3:
                self._add_suggestion(Suggestion(
                    title=f"tool ({tool}) has failed {cf} times consecutively",
                    detail=f"success_rate={v['success_rate']}\n"
                           f"average_latency={v['average_latency']:.3f}ms\n"
                           f"circuit_state={v['circuit_state']}",
                    action="1.check whether the tool dependency service is functioning properly;\n"
                           "2.view error log;\n"
                           "3.consider increasing the timeout period or implementing a fallback strategy.",
                    priority=9
                ))

        updater = getattr(self._direct, "update_routing_penalties", None)
        if updater: updater(routing_penalties)
        self._generate_routing_suggestions(agent_state)

    @staticmethod
    def _routing_penalty(success_rate: float, average_latency: float) -> float:
        penalty = 0.0
        if success_rate < 0.9:
            penalty += min(0.5, (0.9 - success_rate) * 2)
        if average_latency > 3000:
            penalty += min(0.4, (average_latency - 3000) / 10000)
        return min(penalty, 0.9)

    def _check_threshold(self, metric: str, value: float, label: str):
        if metric not in _threshold: return
        op, threshold, severity = _threshold[metric]
        op = op.lower()
        triggered = ('gt' == op and value > threshold) or \
                  ('lt' == op and value < threshold)
        if triggered:
            alert = Alert(
                severity=severity,
                metric=f"{metric}:{label}",
                message=f"[{label}:{metric}] {value:.3f} {op} {threshold}",
                value=value,
                threshold=threshold
            )
            self._alerts.append(alert)
            self.warning(f"[{severity.value.upper()} {alert.message}")
            if self._webhook_url:
                asyncio.create_task(self._send_webhook(alert))

    def _add_suggestion(self, suggestion: Suggestion):
        if not any(s.title == suggestion.title for s in self._suggestions):
            self._suggestions.append(suggestion)
            self.info(f"suggestion [P{suggestion.priority}] {suggestion.title}")

    def _generate_routing_suggestions(self, agent_states: Dict[str, Any]):
        for agent, v in agent_states.items():
            if v["success_rate"] < 0.5 and v["total"] > 10:
                self._add_suggestion(Suggestion(
                    title=f"agent {agent} low success rate",
                    detail=f"success_rate={v['success_rate']:.3f}\n"
                           f"routing_score={v['routing_score']:.3f}",
                    action=f"direct {self._direct._name} has reduced the routing weight of this agent\n"
                           f"suggestion:\n"
                           f"1.check whether system_prompt needs to be optimized;\n"
                           f"2.check whether the complexity of this type of problem exceeds the capabilities of the agent\n"
                           f"3.consider adding instances of the same type of agent.",
                    priority=8
                ))

    async def _send_webhook(self, alert: Alert):
        try:
            async with AsyncClient(timeout=5.0) as c:
                await c.post(self._webhook_url, json=asdict(alert))
        except Exception as e:
            self.exception(f"failed to send webhook because of exception: {e}")


if __name__ == "__main__":
    # m = Monitor(1, 2)

    ...





