import sqlite3
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import time

# # 添加上一层目录到路径
# parent_dir = Path(__file__).parent.parent
# sys.path.insert(0, str(parent_dir))

# 导入原有的推荐方法和模型
from models.analysis import StockRecommendationInput, RecommendationCriteria, StockScore

# 导入原有的推荐函数（假设在analysis_tools.py中）
from tools.ak_stock_tools import recommend_a_stocks


class SimpleStockDB:
    def __init__(self, db_path: str = "data/stock_recommendations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

    def init_database(self):
        """初始化数据库表（只用一张表）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 股票推荐数据表（一只股票一行）
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                preset_name TEXT NOT NULL,
                analysis_date TEXT NOT NULL,

                -- 基础数据
                current_price REAL,
                change_percent REAL,
                volume INTEGER,
                market_cap REAL,
                pe_ratio REAL,
                pb_ratio REAL,

                -- 技术指标
                rsi_14 REAL,
                sma_20 REAL,
                macd_line REAL,
                macd_signal REAL,
                macd_histogram REAL,
                macd_signal_type TEXT,
                sma_position TEXT,
                volume_ratio REAL,

                -- 评分
                technical_score REAL,
                fundamental_score REAL,
                overall_score REAL,
                recommendation_reasons TEXT,  -- JSON格式

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, market_type, preset_name, analysis_date)
            )
            """)

            # 创建索引
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_preset_date ON stock_recommendations(preset_name, market_type, analysis_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol ON stock_recommendations(symbol, market_type)"
            )

            conn.commit()

    def save_recommendations(self, recommendations: List[Dict]):
        """保存推荐数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if recommendations:
                # 获取当天日期
                analysis_date = recommendations[0]["analysis_date"]

                # 先清空当天的数据
                print(f"清空{analysis_date}的旧数据...")
                cursor.execute(
                    "DELETE FROM stock_recommendations WHERE analysis_date = ?",
                    (analysis_date,),
                )

                # 插入新数据
                for rec in recommendations:
                    cursor.execute(
                        """
                    INSERT INTO stock_recommendations
                    (symbol, name, market_type, preset_name, analysis_date, current_price, change_percent,
                     volume, market_cap, pe_ratio, pb_ratio, rsi_14, sma_20, macd_line, macd_signal,
                     macd_histogram, macd_signal_type, sma_position, volume_ratio, technical_score,
                     fundamental_score, overall_score, recommendation_reasons)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            rec["symbol"],
                            rec["name"],
                            rec["market_type"],
                            rec["preset_name"],
                            rec["analysis_date"],
                            rec["current_price"],
                            rec["change_percent"],
                            rec["volume"],
                            rec["market_cap"],
                            rec["pe_ratio"],
                            rec["pb_ratio"],
                            rec["rsi_14"],
                            rec["sma_20"],
                            rec["macd_line"],
                            rec["macd_signal"],
                            rec["macd_histogram"],
                            rec["macd_signal_type"],
                            rec["sma_position"],
                            rec["volume_ratio"],
                            rec["technical_score"],
                            rec["fundamental_score"],
                            rec["overall_score"],
                            json.dumps(
                                rec["recommendation_reasons"], ensure_ascii=False
                            ),
                        ),
                    )

            conn.commit()

    def get_recommendations(
        self, preset_name: str, market_type: str, limit: int = 20
    ) -> List[Dict]:
        """获取推荐结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT * FROM stock_recommendations
            WHERE preset_name = ? AND market_type = ?
            ORDER BY analysis_date DESC, overall_score DESC
            LIMIT ?
            """,
                (preset_name, market_type, limit),
            )

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            results = []
            for row in rows:
                rec_dict = dict(zip(columns, row))
                if rec_dict["recommendation_reasons"]:
                    rec_dict["recommendation_reasons"] = json.loads(
                        rec_dict["recommendation_reasons"]
                    )
                results.append(rec_dict)

            return results

    def get_latest_analysis_date(self, market_type: str) -> Optional[str]:
        """获取最新分析日期"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT MAX(analysis_date) FROM stock_recommendations
            WHERE market_type = ?
            """,
                (market_type,),
            )

            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM stock_recommendations")
            total_records = cursor.fetchone()[0]

            # 按市场统计
            cursor.execute("""
            SELECT market_type, COUNT(*) as count, MAX(analysis_date) as latest_date
            FROM stock_recommendations
            GROUP BY market_type
            """)
            market_stats = cursor.fetchall()

            # 按策略统计
            cursor.execute("""
            SELECT preset_name, COUNT(*) as count
            FROM stock_recommendations
            GROUP BY preset_name
            """)
            preset_stats = cursor.fetchall()

            return {
                "total_records": total_records,
                "market_stats": [
                    {"market_type": row[0], "count": row[1], "latest_date": row[2]}
                    for row in market_stats
                ],
                "preset_stats": [
                    {"preset_name": row[0], "count": row[1]} for row in preset_stats
                ],
            }


def stock_score_to_dict(
    stock_score: StockScore, market_type: str, preset_name: str, analysis_date: str
) -> Dict:
    """将StockScore对象转换为数据库记录字典"""
    return {
        "symbol": stock_score.symbol,
        "name": stock_score.name,
        "market_type": market_type,
        "preset_name": preset_name,
        "analysis_date": analysis_date,
        "current_price": stock_score.current_price,
        "change_percent": stock_score.change_percent,
        "volume": 0,  # 从StockScore中没有volume字段，设为0
        "market_cap": stock_score.market_cap,
        "pe_ratio": stock_score.pe_ratio,
        "pb_ratio": stock_score.pb_ratio,
        "rsi_14": stock_score.rsi,
        "sma_20": 0,  # 可以根据需要从其他地方获取
        "macd_line": 0,  # 可以根据需要从其他地方获取
        "macd_signal": 0,  # 可以根据需要从其他地方获取
        "macd_histogram": 0,  # 可以根据需要从其他地方获取
        "macd_signal_type": stock_score.macd_signal,
        "sma_position": stock_score.sma_position,
        "volume_ratio": stock_score.volume_ratio,
        "technical_score": stock_score.technical_score,
        "fundamental_score": stock_score.fundamental_score,
        "overall_score": stock_score.overall_score,
        "recommendation_reasons": stock_score.recommendation_reason,
    }


class MockContext:
    """模拟FastMCP Context用于脚本运行"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    async def info(self, message: str):
        if self.verbose:
            print(f"INFO: {message}")

    async def warning(self, message: str):
        if self.verbose:
            print(f"WARNING: {message}")

    async def error(self, message: str):
        if self.verbose:
            print(f"ERROR: {message}")


async def run_analysis_and_save(db: SimpleStockDB, verbose: bool = True):
    """运行分析并保存到数据库"""
    ctx = MockContext(verbose)
    analysis_date = datetime.now().strftime("%Y-%m-%d")

    print(f"开始运行股票分析并保存到数据库 - {analysis_date}")

    # 定义推荐策略
    presets = {
        "value_stocks": RecommendationCriteria(
            pe_ratio_min=5, pe_ratio_max=20, rsi_min=30, rsi_max=60, market_cap_min=50
        ),
        "growth_stocks": RecommendationCriteria(
            rsi_min=40,
            rsi_max=75,
            require_above_sma=True,
            price_change_min=-2,
            price_change_max=8,
        ),
        "oversold_bounce": RecommendationCriteria(
            rsi_min=20, rsi_max=35, price_change_min=-10, price_change_max=2
        ),
        "momentum_stocks": RecommendationCriteria(
            require_golden_cross=True, require_above_sma=True, rsi_min=50, rsi_max=80
        ),
        "large_cap_stable": RecommendationCriteria(
            market_cap_min=500, pe_ratio_min=8, pe_ratio_max=25, rsi_min=35, rsi_max=65
        ),
    }

    all_recommendations = []

    # 分析A股市场
    print("\n=== 开始分析A股市场 ===")
    for preset_name, criteria in presets.items():
        print(f"\n处理A股策略: {preset_name}")

        try:
            input_params = StockRecommendationInput(
                market_type="a_stock", criteria=criteria, limit=20, timeframe="1d"
            )

            result = await recommend_a_stocks(ctx, input_params)
            time.sleep(5)

            if result.error:
                print(f"A股{preset_name}策略失败: {result.error}")
                continue

            print(f"A股{preset_name}策略完成，推荐{len(result.recommendations)}只股票")

            # 转换为数据库记录
            for stock_score in result.recommendations:
                rec_dict = stock_score_to_dict(
                    stock_score, "a_stock", preset_name, analysis_date
                )
                all_recommendations.append(rec_dict)

        except Exception as e:
            print(f"A股{preset_name}策略出错: {e}")
            continue

    # # 分析港股市场
    # print("\n=== 开始分析港股市场 ===")
    # for preset_name, criteria in presets.items():
    #     print(f"\n处理港股策略: {preset_name}")

    #     try:
    #         input_params = StockRecommendationInput(
    #             market_type="hk_stock",
    #             criteria=criteria,
    #             limit=20,
    #             timeframe="1d"
    #         )

    #         result = await recommend_hk_stocks(ctx, input_params)

    #         if result.error:
    #             print(f"港股{preset_name}策略失败: {result.error}")
    #             continue

    #         print(f"港股{preset_name}策略完成，推荐{len(result.recommendations)}只股票")

    #         # 转换为数据库记录
    #         for stock_score in result.recommendations:
    #             rec_dict = stock_score_to_dict(stock_score, "hk_stock", preset_name, analysis_date)
    #             all_recommendations.append(rec_dict)

    #     except Exception as e:
    #         print(f"港股{preset_name}策略出错: {e}")
    #         continue

    # 保存到数据库
    if all_recommendations:
        print(f"\n保存{len(all_recommendations)}条推荐记录到数据库...")
        db.save_recommendations(all_recommendations)
        print("保存完成!")
    else:
        print("\n没有推荐数据需要保存")

    return len(all_recommendations)


async def query_recommendations(
    db: SimpleStockDB, preset_name: str, market_type: str, limit: int = 10
):
    """查询推荐结果"""
    print(f"\n=== 查询 {market_type} - {preset_name} 推荐结果 ===")

    results = db.get_recommendations(preset_name, market_type, limit)

    if not results:
        print("未找到推荐数据")
        return

    print(f"找到 {len(results)} 条推荐记录:")
    print()

    for i, rec in enumerate(results, 1):
        reasons = (
            ", ".join(rec["recommendation_reasons"])
            if rec["recommendation_reasons"]
            else "无"
        )

        print(f"{i}. {rec['name']}({rec['symbol']})")
        print(f"   价格: {rec['current_price']:.2f} ({rec['change_percent']:+.2f}%)")
        print(
            f"   评分: 技术{rec['technical_score']:.1f} 基本面{rec['fundamental_score']:.1f} 综合{rec['overall_score']:.1f}"
        )
        print(f"   RSI: {rec['rsi_14'] or 'N/A'}")
        print(f"   MACD信号: {rec['macd_signal_type'] or 'N/A'}")
        print(f"   SMA位置: {rec['sma_position'] or 'N/A'}")
        print(f"   推荐理由: {reasons}")
        print(f"   分析日期: {rec['analysis_date']}")
        print()


def show_database_stats(db: SimpleStockDB):
    """显示数据库统计信息"""
    print("\n=== 数据库统计信息 ===")

    stats = db.get_stats()

    print(f"总记录数: {stats['total_records']}")

    print("\n按市场统计:")
    for market in stats["market_stats"]:
        print(
            f"  {market['market_type']}: {market['count']} 条记录，最新日期: {market['latest_date']}"
        )

    print("\n按策略统计:")
    for preset in stats["preset_stats"]:
        print(f"  {preset['preset_name']}: {preset['count']} 条记录")


async def main():
    """主函数"""
    print("=== 股票推荐分析系统 ===")

    # 初始化数据库
    db = SimpleStockDB()
    print("数据库初始化完成")

    # 显示当前数据库状态
    show_database_stats(db)

    while True:
        print("\n请选择操作:")
        print("1. 运行完整分析并保存到数据库")
        print("2. 查询A股价值股推荐")
        print("3. 查询A股成长股推荐")
        print("4. 查询A股超卖反弹推荐")
        print("5. 查询A股动量股推荐")
        print("6. 查询A股大盘稳健股推荐")
        print("7. 查询港股价值股推荐")
        print("8. 查询港股成长股推荐")
        print("9. 显示数据库统计")
        print("10. 自定义查询")
        print("0. 退出")

        choice = input("\n请输入选择 (0-10): ").strip()

        if choice == "0":
            print("再见!")
            break
        elif choice == "1":
            saved_count = await run_analysis_and_save(db)
            print(f"\n分析完成，共保存 {saved_count} 条推荐记录")
            show_database_stats(db)
        elif choice == "2":
            await query_recommendations(db, "value_stocks", "a_stock")
        elif choice == "3":
            await query_recommendations(db, "growth_stocks", "a_stock")
        elif choice == "4":
            await query_recommendations(db, "oversold_bounce", "a_stock")
        elif choice == "5":
            await query_recommendations(db, "momentum_stocks", "a_stock")
        elif choice == "6":
            await query_recommendations(db, "large_cap_stable", "a_stock")
        elif choice == "7":
            await query_recommendations(db, "value_stocks", "hk_stock")
        elif choice == "8":
            await query_recommendations(db, "growth_stocks", "hk_stock")
        elif choice == "9":
            show_database_stats(db)
        elif choice == "10":
            print(
                "\n可用策略: value_stocks, growth_stocks, oversold_bounce, momentum_stocks, large_cap_stable"
            )
            print("可用市场: a_stock, hk_stock")
            preset = input("请输入策略名称: ").strip()
            market = input("请输入市场类型: ").strip()
            limit = input("请输入返回数量 (默认10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            await query_recommendations(db, preset, market, limit)
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    print("启动股票推荐分析系统...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback

        traceback.print_exc()
