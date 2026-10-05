"""
游戏内玩家留存分析与生命周期系统 - 核心模块

功能：
- retention_analysis: 留存率（次日/7日/30日）+ 按注册日期 cohort 分组分析
- lifecycle: 玩家生命周期阶段（新手/成长/成熟/衰退/流失）与阶段转换记录
- churn_prediction: 基于特征的逻辑回归模型预测玩家流失概率
- recall_strategy: 按流失程度分层的召回策略与效果评估
- player_health: 玩家健康度指标（登录频率/游戏时长/消费金额）与预警
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time

SECONDS_PER_DAY = 86400

# 生命周期阶段
STAGE_NEWBIE = "newbie"      # 新手
STAGE_GROWTH = "growth"      # 成长
STAGE_MATURE = "mature"      # 成熟
STAGE_DECLINE = "decline"    # 衰退
STAGE_CHURNED = "churned"    # 流失

STAGE_NAMES = {
    STAGE_NEWBIE: "新手",
    STAGE_GROWTH: "成长",
    STAGE_MATURE: "成熟",
    STAGE_DECLINE: "衰退",
    STAGE_CHURNED: "流失",
}

# 生命周期阶段阈值（天）
NEWBIE_DAYS = 7          # 注册 7 天内为新手
GROWTH_DAYS = 30         # 注册 30 天内为成长
DECLINE_INACTIVE_DAYS = 7    # 7 天未登录进入衰退
CHURN_INACTIVE_DAYS = 30     # 30 天未登录判定流失

# 健康度阈值
HEALTH_WARNING_SCORE = 70
HEALTH_CRITICAL_SCORE = 40

# 流失预测模型（逻辑回归）权重：基于标准化特征
_CHURN_WEIGHTS = {
    "days_since_last_login": 0.12,
    "login_frequency": -1.6,      # 近7天登录天数占比
    "avg_session_hours": -0.9,    # 平均每次游戏时长（小时）
    "daily_spending": -0.5,       # 日均消费
}
_CHURN_BIAS = -0.8

# 召回分层阈值（未登录天数）
RECALL_TIERS = [
    ("轻度流失", 7, 15),
    ("中度流失", 15, 30),
    ("重度流失", 30, None),
]

# 各分层预期召回率与单次召回成本（用于效果评估）
_TIER_EXPECTED_RECALL_RATE = {"轻度流失": 0.45, "中度流失": 0.25, "重度流失": 0.08}
_TIER_RECALL_COST = {"轻度流失": 2.0, "中度流失": 5.0, "重度流失": 10.0}


@dataclass
class Config:
    pass


def _now() -> float:
    return time.time()


def _days_between(start: float, end: float) -> float:
    return max(0.0, (end - start) / SECONDS_PER_DAY)


def _register_date(register_time: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(register_time))


class PlayerRetention:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        # 玩家生命周期阶段转换记录: {player_id: {"stage": str, "transitions": [...]}}
        self._lifecycle: Dict[str, Dict] = {}
        # 召回执行记录: [{"player_id", "tier", "time", "recalled"}]
        self._recall_records: List[Dict] = []

    # ------------------------------------------------------------------
    # 框架接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._lifecycle = {}
        self._recall_records = []

    # ------------------------------------------------------------------
    # 玩家特征
    # ------------------------------------------------------------------
    def _extract_features(self, player: Dict, current_time: float) -> Dict[str, float]:
        register_time = player.get("register_time", current_time)
        last_login_time = player.get("last_login_time", register_time)
        days_since_register = _days_between(register_time, current_time)
        days_since_last_login = _days_between(last_login_time, current_time)

        login_days = player.get("login_days")
        if login_days is None:
            login_days = len(player.get("login_history", []) or [])
        recent_window = min(max(days_since_register, 1.0), 7.0)
        login_frequency = min(1.0, login_days / recent_window) if recent_window > 0 else 0.0

        total_play_time = float(player.get("total_play_time", 0.0))  # 小时
        sessions = max(float(player.get("sessions", login_days or 1)), 1.0)
        avg_session_hours = total_play_time / sessions

        total_spent = float(player.get("total_spent", 0.0))
        daily_spending = total_spent / max(days_since_register, 1.0)

        return {
            "days_since_register": days_since_register,
            "days_since_last_login": days_since_last_login,
            "login_frequency": login_frequency,
            "avg_session_hours": avg_session_hours,
            "daily_spending": daily_spending,
            "total_play_time": total_play_time,
            "total_spent": total_spent,
        }

    # ------------------------------------------------------------------
    # 1. 留存分析：留存率 + cohort
    # ------------------------------------------------------------------
    def retention_analysis(
        self,
        players: List[Dict],
        current_time: Optional[float] = None,
    ) -> Dict:
        """计算整体留存率（次日/7日/30日）并按注册日期做 cohort 分组分析。"""
        current_time = current_time if current_time is not None else _now()
        cohorts: Dict[str, List[Dict]] = {}
        for player in players:
            key = _register_date(player.get("register_time", current_time))
            cohorts.setdefault(key, []).append(player)

        cohort_report = {}
        for date_key in sorted(cohorts):
            members = cohorts[date_key]
            cohort_report[date_key] = {
                "size": len(members),
                "retention": self._cohort_retention(members, current_time),
            }

        return {
            "total_players": len(players),
            "online_count": sum(1 for p in players if p.get("online", False)),
            "retention_rates": self._cohort_retention(players, current_time),
            "cohorts": cohort_report,
        }

    def _cohort_retention(self, players: List[Dict], current_time: float) -> Dict[str, float]:
        """day N 留存率 = 注册满 N 天的玩家中，第 N 天之后仍活跃的比例。"""
        rates = {}
        for label, days in (("day1", 1), ("day7", 7), ("day30", 30)):
            eligible = [
                p for p in players
                if _days_between(p.get("register_time", current_time), current_time) >= days
            ]
            if not eligible:
                rates[label] = 0.0
                continue
            retained = sum(
                1 for p in eligible
                if _days_between(
                    p.get("register_time", current_time),
                    p.get("last_login_time", p.get("register_time", current_time)),
                ) >= days
            )
            rates[label] = round(retained / len(eligible), 4)
        return rates

    # ------------------------------------------------------------------
    # 2. 生命周期：阶段 + 转换
    # ------------------------------------------------------------------
    def _classify_stage(self, features: Dict[str, float]) -> str:
        if features["days_since_last_login"] >= CHURN_INACTIVE_DAYS:
            return STAGE_CHURNED
        if features["days_since_last_login"] >= DECLINE_INACTIVE_DAYS:
            return STAGE_DECLINE
        if features["days_since_register"] <= NEWBIE_DAYS:
            return STAGE_NEWBIE
        if features["days_since_register"] <= GROWTH_DAYS:
            return STAGE_GROWTH
        return STAGE_MATURE

    def lifecycle(
        self,
        player: Dict,
        current_time: Optional[float] = None,
    ) -> Dict:
        """返回玩家当前生命周期阶段，并记录阶段转换历史。"""
        current_time = current_time if current_time is not None else _now()
        player_id = str(player.get("player_id", "unknown"))
        features = self._extract_features(player, current_time)
        stage = self._classify_stage(features)

        record = self._lifecycle.setdefault(
            player_id, {"stage": None, "transitions": []}
        )
        previous = record["stage"]
        if previous != stage:
            if previous is not None:
                record["transitions"].append({
                    "from": previous,
                    "to": stage,
                    "from_name": STAGE_NAMES[previous],
                    "to_name": STAGE_NAMES[stage],
                    "time": current_time,
                })
            record["stage"] = stage
            record["stage_since"] = current_time

        order = [STAGE_NEWBIE, STAGE_GROWTH, STAGE_MATURE, STAGE_DECLINE, STAGE_CHURNED]
        idx = order.index(stage)
        next_stage = order[idx + 1] if idx + 1 < len(order) else None

        return {
            "player_id": player_id,
            "stage": stage,
            "stage_name": STAGE_NAMES[stage],
            "days_in_stage": round(_days_between(record.get("stage_since", current_time), current_time), 2),
            "next_stage": next_stage,
            "next_stage_name": STAGE_NAMES[next_stage] if next_stage else None,
            "transitions": list(record["transitions"]),
        }

    # ------------------------------------------------------------------
    # 3. 流失预测：特征 + 模型
    # ------------------------------------------------------------------
    def churn_prediction(
        self,
        player: Dict,
        current_time: Optional[float] = None,
    ) -> Dict:
        """用逻辑回归模型基于玩家行为特征预测流失概率。"""
        current_time = current_time if current_time is not None else _now()
        features = self._extract_features(player, current_time)

        z = _CHURN_BIAS
        contributions = {}
        for name, weight in _CHURN_WEIGHTS.items():
            value = features[name]
            # 对量级较大的特征做压缩，避免数值爆炸
            if name in ("days_since_last_login",):
                value = min(value, 60.0)
            elif name in ("avg_session_hours",):
                value = min(value, 10.0)
            elif name in ("daily_spending",):
                value = min(value, 100.0)
            contribution = weight * value
            contributions[name] = round(contribution, 4)
            z += contribution

        probability = 1.0 / (1.0 + math.exp(-z))
        if probability >= 0.7:
            risk_level = "high"
        elif probability >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "player_id": str(player.get("player_id", "unknown")),
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "model": "logistic_regression",
            "features": {
                "days_since_last_login": round(features["days_since_last_login"], 2),
                "login_frequency": round(features["login_frequency"], 4),
                "avg_session_hours": round(features["avg_session_hours"], 2),
                "daily_spending": round(features["daily_spending"], 2),
            },
            "feature_contributions": contributions,
        }

    # ------------------------------------------------------------------
    # 4. 召回策略：分层 + 效果评估
    # ------------------------------------------------------------------
    def _recall_tier(self, days_since_last_login: float) -> Optional[str]:
        for name, low, high in RECALL_TIERS:
            if days_since_last_login >= low and (high is None or days_since_last_login < high):
                return name
        return None

    def recall_strategy(
        self,
        player: Dict,
        current_time: Optional[float] = None,
    ) -> Dict:
        """按流失程度分层生成召回策略，并附带效果评估。"""
        current_time = current_time if current_time is not None else _now()
        features = self._extract_features(player, current_time)
        days_inactive = features["days_since_last_login"]
        tier = self._recall_tier(days_inactive)

        if tier is None:
            return {
                "player_id": str(player.get("player_id", "unknown")),
                "need_recall": False,
                "tier": None,
                "days_inactive": round(days_inactive, 2),
                "strategies": [],
                "evaluation": None,
            }

        strategies_by_tier = {
            "轻度流失": ["推送登录提醒", "发放小额登录奖励", "推送未读活动通知"],
            "中度流失": ["发放回归礼包", "专属客服触达", "推送版本更新亮点", "限时回归任务"],
            "重度流失": ["高额回归礼包", "电话/短信召回", "老玩家专属回归活动", "账号资产保留提醒"],
        }
        strategies = strategies_by_tier[tier]
        expected_rate = _TIER_EXPECTED_RECALL_RATE[tier]
        cost = _TIER_RECALL_COST[tier]
        player_value = max(features["daily_spending"] * 30.0, 1.0)
        expected_roi = round(expected_rate * player_value / cost, 4)

        return {
            "player_id": str(player.get("player_id", "unknown")),
            "need_recall": True,
            "tier": tier,
            "days_inactive": round(days_inactive, 2),
            "strategies": strategies,
            "evaluation": {
                "expected_recall_rate": expected_rate,
                "recall_cost": cost,
                "expected_roi": expected_roi,
                "actual_recall_rate": self._actual_recall_rate(tier),
            },
        }

    def record_recall_result(self, player_id: str, tier: str, recalled: bool,
                             record_time: Optional[float] = None):
        """记录一次召回执行结果，用于效果评估。"""
        self._recall_records.append({
            "player_id": str(player_id),
            "tier": tier,
            "recalled": bool(recalled),
            "time": record_time if record_time is not None else _now(),
        })

    def _actual_recall_rate(self, tier: str) -> Optional[float]:
        records = [r for r in self._recall_records if r["tier"] == tier]
        if not records:
            return None
        return round(sum(1 for r in records if r["recalled"]) / len(records), 4)

    # ------------------------------------------------------------------
    # 5. 玩家健康度：指标 + 预警
    # ------------------------------------------------------------------
    def player_health(
        self,
        player: Dict,
        current_time: Optional[float] = None,
    ) -> Dict:
        """综合登录频率、游戏时长、消费金额计算健康度并给出预警。"""
        current_time = current_time if current_time is not None else _now()
        features = self._extract_features(player, current_time)

        metrics = {
            "login_frequency": round(features["login_frequency"], 4),
            "avg_session_hours": round(features["avg_session_hours"], 2),
            "daily_spending": round(features["daily_spending"], 2),
            "days_since_last_login": round(features["days_since_last_login"], 2),
        }

        # 各指标归一化到 [0, 1] 后加权求和
        login_score = features["login_frequency"]
        play_score = min(features["avg_session_hours"] / 3.0, 1.0)
        spend_score = min(features["daily_spending"] / 10.0, 1.0)
        recency_score = max(0.0, 1.0 - features["days_since_last_login"] / CHURN_INACTIVE_DAYS)

        score = round(100.0 * (
            0.35 * login_score +
            0.20 * play_score +
            0.15 * spend_score +
            0.30 * recency_score
        ), 2)

        warnings = []
        if features["login_frequency"] < 0.3:
            warnings.append("登录频率过低")
        if features["avg_session_hours"] < 0.5:
            warnings.append("游戏时长过短")
        if features["days_since_last_login"] >= DECLINE_INACTIVE_DAYS:
            warnings.append("长时间未登录")
        if score < HEALTH_CRITICAL_SCORE:
            warnings.append("健康度严重偏低，存在流失风险")

        if score < HEALTH_CRITICAL_SCORE:
            level = "critical"
        elif score < HEALTH_WARNING_SCORE:
            level = "warning"
        else:
            level = "healthy"

        return {
            "player_id": str(player.get("player_id", "unknown")),
            "score": score,
            "level": level,
            "metrics": metrics,
            "warnings": warnings,
            "alert": level != "healthy",
        }
