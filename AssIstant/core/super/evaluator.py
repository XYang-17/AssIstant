from typing import List, Dict, Any, Optional
from statistics import mean
from anthropic import AsyncAnthropic
from pathlib import Path
from json import loads, dumps
from dataclasses import asdict
from datetime import datetime

from base.evaluator import *
from AssIstant.core.agent.interpreter import IntentInterpreter
from AssIstant.core.super.director import Director, AgentRequest
from AssIstant.common.logger import FormatLog
from AssIstant.common.config import EVALUATE_PASS_THRESHOLD
from AssIstant.common.utils import truncate_text


class IntentEvaluator:
    def __init__(self, interpreter: IntentInterpreter):
        self._interpreter = interpreter

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        pd, gt,details = [], [], []
        for case in cases:
            res = await self._interpreter.interpret(case.message)
            predicted = res.intent
            pd.append(predicted)
            gt.append(case.expected_intent)
            details.append({
                "message": case.message,
                "expected": case.expected_intent.name(),
                "predicted": predicted.name(),
                "confidence": res.confidence,
                "reasoning": res.reasoning
            })

        correct = sum(p == g for p, g in zip(pd, gt))
        accuracy = correct / len(pd) if pd else 0.0

        labels = sorted(set(gt + pd))
        indicator_class = {}
        for label in labels:
            tp, fp, fn = 0, 0, 0
            for p, g in zip(pd, gt):
                if p == g:
                    if p == label: tp += 1
                else:
                    if p == label: fp += 1
                    elif g == label: fn += 1
            pre_class = tp / (tp + fp) if (tp + fp) else 0.0
            rec_class = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * pre_class * rec_class / (pre_class + rec_class) if (pre_class + rec_class) else 0.0
            indicator_class[label] = {"precision": pre_class, "recall": rec_class, "F1": f1}

        macro_f1 = mean(c["F1"] for c in indicator_class.values()) if indicator_class else 0.0

        return {
            "total": len(cases),
            "accuracy": round(accuracy, 3),
            "macro_F1": round(macro_f1, 3),
            "correct": correct,
            "indicator_per_class": indicator_class,
            "cases": details
        }

class E2EEvaluator(FormatLog):
    def __init__(
        self,
        director: Director,
        interpreter: IntentInterpreter,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
        name: str = "e2eEvaluator"
    ):
        FormatLog.__init__(self, name)
        client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._director = director
        self._judge = Judge(client, model)
        self._intent_evaluator = IntentEvaluator(interpreter)
        self._baseline_path = Path(baseline_path) if baseline_path else None

        self._baseline: Optional[EvalReport] = self._load_baseline()
        self._history: List[EvalReport] = list()

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    async def run(
       self,
        intent_cases: Optional[List[IntentTestCase]] = None,
        dialog_cases: Optional[List[Dict[str, Any]]] = None
    ) -> EvalReport:
        results = list()
        all_scores = { score: [] for score in QualityScores.all_scores()}

        intent_metrics = dict()
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= EVALUATE_PASS_THRESHOLD
            info = {"accuracy": intent_metrics['accuracy'], "macro_F1": intent_metrics['macro_F1']}
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores=info,
                detail=str(info),
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", [])
                }
            ))

        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        average_scores = {k: round(mean(v), 3) for k, v in all_scores.items() if v}
        if intent_metrics: average_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(passed_count / len(results) if results else 0.0, 3),
            average_scores=average_scores,
            regressions=self._regressions(average_scores),
            recommendations=self._recommendations(average_scores),
            results=results
        )

        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        questions = self._dialog_turns(case)
        if not questions: return []

        user_id = str(case.get("user_id") or "eval_user")
        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        history: List[Dict[str, str]] = list()
        results: List[EvalResult] = list()

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            request = AgentRequest(
                content=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None
            )
            director_response = await self._director.run(request)
            response = director_response.response

            scores = await self._judge.judge(question, response, context or None)
            passed = scores.overall >= EVALUATE_PASS_THRESHOLD

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall
                },
                detail=f"question: {truncate_text(question, 20)}\n"
                       f"response: {truncate_text(response, 20)}\n"
                       f"composite score: {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": response,
                    "agent_type": director_response.agent_type.value.name,
                    "intent": director_response.intent.name() if director_response.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.failed,
                    "judge_error": scores.error
                }
            ))
        return results

    def _regressions(self, current: Dict[str, float]) -> List[str]:
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None: return []
        prev_average_scores = prev_report.average_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev_average_scores and (p_value := prev_average_scores[metric]) > 0:
                delta = (value - p_value) / p_value
                if delta < -0.05:
                    regressions.append(
                        f"{{"
                        f"\"metric\": {metric}, "
                        f"\"lastest\": {p_value:.3f}, "
                        f"\"current\":{value:.3f}, "
                        f"\"regression\":{abs(delta):.3%}"
                        f"}}"
                    )
        return regressions

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = loads(self._baseline_path.read_text(encoding="utf-8"))
            return EvalReport(
                timestamp=data.get("timestamp", ""),
                total=int(data.get("total", 0)),
                passed=int(data.get("passed", 0)),
                pass_rate=float(data.get("pass_rate", 0.0)),
                average_scores=dict(data.get("average_scores", dict())),
                regressions=list(data.get("regressions", list())),
                recommendations=list(data.get("recommendations", list())),
                results=[
                    EvalResult(
                        test_id=res.get("test_id", ""),
                        passed=bool(res.get("passed", False)),
                        scores=dict(res.get("scores", dict())),
                        detail=res.get("detail", ""),
                        metadata=dict(res.get("metadata", dict()))
                    )
                    for res in data.get("results", [])
                ]
            )
        except Exception as e:
            self.exception(f"failed to load baseline because of exception: {e}")

    def _save_baseline(self, report: EvalReport):
        if not self._baseline_path: return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self._baseline = report
        except Exception as e:
            self.exception(f"failed to save baseline because of exception: {e}")

    @staticmethod
    def _recommendations(scores: Dict[str, float]) -> List[str]:
        recommendations = []
        if scores.get("intent_accuracy", 1.0) < 0.9:
            recommendations.append(
                "The current intent interpretation accuracy is less than 90%. "
                "Please increase the number of few-shot examples, "
                "or supplement training data for intent categories with low F1 scores."
            )
        if scores.get("relevance", 1.0) < 0.75:
            recommendations.append(
                "The current response is lacking in relevance, "
                "Please check the Agent system prompt to ensure that the Agent is focused on the user's problem."
            )
        if scores.get("completeness", 1.0) < 0.75:
            recommendations.append(
                "The current response is lacking in completeness, "
                "and the agent may have prematurely ended the answer. "
                "Consider requiring a complete solution in the prompt."
            )
        if scores.get("helpfulness", 1.0) < 0.75:
            recommendations.append(
                "The current response is of low usefulness, "
                "and the answer may be too abstract. "
                "Consider asking the Agent to provide specific operational steps."
            )
        if not recommendations:
            recommendations.append("All indicators have met the standards.")
        return recommendations

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, List):
            return [t for turn in turns if (t := str(turn).strip())]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history: return ""
        lines = [f"{msg['role']}: {msg['content']}" for msg in history[-8:]]
        return f"[context history]\n" + "\n".join(lines)

