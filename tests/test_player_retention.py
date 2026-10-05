"""
游戏内玩家留存分析与生命周期系统 - 验收测试
运行方式：python -m pytest tests/test_player_retention.py -q
共 14 个测试用例
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.retention import PlayerRetention

DAY = 86400
NOW = 1_700_000_000.0  # 固定基准时间，保证测试可重复


def make_player(player_id, register_days_ago, last_login_days_ago,
                login_days=7, total_play_time=20.0, total_spent=100.0,
                sessions=None, online=False):
    return {
        "player_id": player_id,
        "register_time": NOW - register_days_ago * DAY,
        "last_login_time": NOW - last_login_days_ago * DAY,
        "login_days": login_days,
        "total_play_time": total_play_time,
        "total_spent": total_spent,
        "sessions": sessions if sessions is not None else max(login_days, 1),
        "online": online,
    }


class TestPlayerRetention:
    def setup_method(self):
        self.system = PlayerRetention()

    # ---- 留存分析：留存率 + cohort ----

    def test_case_01(self):
        """留存分析返回次日/7日/30日留存率"""
        players = [make_player(i, 40, 1) for i in range(10)]
        result = self.system.retention_analysis(players, current_time=NOW)
        rates = result["retention_rates"]
        assert set(rates.keys()) == {"day1", "day7", "day30"}
        assert all(0.0 <= v <= 1.0 for v in rates.values())

    def test_case_02(self):
        """留存率计算正确：注册40天、最近1天登录的玩家全部留存"""
        retained = [make_player(i, 40, 1) for i in range(8)]
        churned = [make_player(100 + i, 40, 40) for i in range(2)]
        result = self.system.retention_analysis(retained + churned, current_time=NOW)
        rates = result["retention_rates"]
        assert rates["day1"] == pytest.approx(0.8, abs=1e-3)
        assert rates["day7"] == pytest.approx(0.8, abs=1e-3)
        assert rates["day30"] == pytest.approx(0.8, abs=1e-3)

    def test_case_03(self):
        """留存分析按注册日期进行 cohort 分组"""
        players = (
            [make_player(i, 40, 1) for i in range(3)]
            + [make_player(10 + i, 20, 2) for i in range(5)]
        )
        result = self.system.retention_analysis(players, current_time=NOW)
        cohorts = result["cohorts"]
        assert len(cohorts) == 2
        sizes = sorted(c["size"] for c in cohorts.values())
        assert sizes == [3, 5]
        for cohort in cohorts.values():
            assert set(cohort["retention"].keys()) == {"day1", "day7", "day30"}

    # ---- 生命周期：阶段 + 转换 ----

    def test_case_04(self):
        """生命周期正确识别五个阶段"""
        cases = [
            (make_player(1, 3, 0), "newbie"),
            (make_player(2, 15, 1), "growth"),
            (make_player(3, 60, 1), "mature"),
            (make_player(4, 60, 10), "decline"),
            (make_player(5, 60, 45), "churned"),
        ]
        for player, expected in cases:
            result = self.system.lifecycle(player, current_time=NOW)
            assert result["stage"] == expected
            assert result["stage_name"] in ("新手", "成长", "成熟", "衰退", "流失")

    def test_case_05(self):
        """生命周期记录阶段转换"""
        player = make_player(1, 3, 0)
        first = self.system.lifecycle(player, current_time=NOW)
        assert first["stage"] == "newbie"
        assert first["transitions"] == []

        # 同一玩家 40 天未登录后应进入流失阶段并产生转换记录
        player["last_login_time"] = NOW - 40 * DAY
        player["register_time"] = NOW - 60 * DAY
        second = self.system.lifecycle(player, current_time=NOW)
        assert second["stage"] == "churned"
        assert len(second["transitions"]) == 1
        transition = second["transitions"][0]
        assert transition["from"] == "newbie"
        assert transition["to"] == "churned"

    def test_case_06(self):
        """生命周期给出下一阶段信息"""
        result = self.system.lifecycle(make_player(1, 3, 0), current_time=NOW)
        assert result["next_stage"] == "growth"
        result = self.system.lifecycle(make_player(2, 60, 45), current_time=NOW)
        assert result["next_stage"] is None  # 流失已是最终阶段

    # ---- 流失预测：特征 + 模型 ----

    def test_case_07(self):
        """流失预测返回 [0,1] 区间的概率与风险等级"""
        result = self.system.churn_prediction(make_player(1, 30, 5), current_time=NOW)
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["risk_level"] in ("low", "medium", "high")

    def test_case_08(self):
        """流失预测包含关键行为特征"""
        result = self.system.churn_prediction(make_player(1, 30, 5), current_time=NOW)
        features = result["features"]
        for key in ("days_since_last_login", "login_frequency",
                    "avg_session_hours", "daily_spending"):
            assert key in features
        assert "feature_contributions" in result

    def test_case_09(self):
        """流失预测使用模型：高流失风险玩家概率显著高于活跃玩家"""
        active = make_player(1, 30, 0, login_days=7, total_play_time=30.0,
                             total_spent=300.0)
        at_risk = make_player(2, 60, 25, login_days=0, total_play_time=0.5,
                              total_spent=0.0)
        p_active = self.system.churn_prediction(active, current_time=NOW)
        p_risk = self.system.churn_prediction(at_risk, current_time=NOW)
        assert p_active["model"] == "logistic_regression"
        assert p_risk["churn_probability"] > p_active["churn_probability"]
        assert p_risk["risk_level"] == "high"
        assert p_active["risk_level"] == "low"

    # ---- 召回策略：分层 + 效果评估 ----

    def test_case_10(self):
        """召回策略按流失程度分层"""
        light = self.system.recall_strategy(make_player(1, 60, 10), current_time=NOW)
        medium = self.system.recall_strategy(make_player(2, 60, 20), current_time=NOW)
        heavy = self.system.recall_strategy(make_player(3, 60, 45), current_time=NOW)
        assert light["tier"] == "轻度流失"
        assert medium["tier"] == "中度流失"
        assert heavy["tier"] == "重度流失"
        # 不同分层使用不同策略
        assert light["strategies"] != heavy["strategies"]
        assert all(len(r["strategies"]) > 0 for r in (light, medium, heavy))

    def test_case_11(self):
        """召回策略包含效果评估，且能根据召回记录计算实际召回率"""
        result = self.system.recall_strategy(make_player(1, 60, 10), current_time=NOW)
        evaluation = result["evaluation"]
        assert 0.0 < evaluation["expected_recall_rate"] <= 1.0
        assert evaluation["recall_cost"] > 0
        assert "expected_roi" in evaluation
        assert evaluation["actual_recall_rate"] is None  # 尚无召回记录

        self.system.record_recall_result("1", "轻度流失", True)
        self.system.record_recall_result("2", "轻度流失", False)
        result = self.system.recall_strategy(make_player(3, 60, 10), current_time=NOW)
        assert result["evaluation"]["actual_recall_rate"] == pytest.approx(0.5)

    def test_case_12(self):
        """活跃玩家不需要召回"""
        result = self.system.recall_strategy(make_player(1, 60, 1), current_time=NOW)
        assert result["need_recall"] is False
        assert result["tier"] is None
        assert result["strategies"] == []

    # ---- 玩家健康度：指标 + 预警 ----

    def test_case_13(self):
        """健康度包含登录频率、游戏时长、消费金额指标与分数"""
        result = self.system.player_health(make_player(1, 30, 1), current_time=NOW)
        metrics = result["metrics"]
        for key in ("login_frequency", "avg_session_hours",
                    "daily_spending", "days_since_last_login"):
            assert key in metrics
        assert 0.0 <= result["score"] <= 100.0
        assert result["level"] in ("healthy", "warning", "critical")

    def test_case_14(self):
        """健康度低的玩家触发预警，活跃玩家不预警"""
        healthy = self.system.player_health(
            make_player(1, 30, 0, login_days=7, total_play_time=30.0,
                        total_spent=300.0),
            current_time=NOW,
        )
        unhealthy = self.system.player_health(
            make_player(2, 60, 20, login_days=0, total_play_time=0.2,
                        total_spent=0.0),
            current_time=NOW,
        )
        assert healthy["level"] == "healthy"
        assert healthy["alert"] is False
        assert unhealthy["level"] in ("warning", "critical")
        assert unhealthy["alert"] is True
        assert len(unhealthy["warnings"]) > 0
        assert unhealthy["score"] < healthy["score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
