"""
线上监控工具
记录用户查询日志、反馈数据，支持定期抽样评估和报告导出
"""

import sys
import os
import json
import time
import sqlite3
import statistics
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR

MONITOR_DB_PATH = Path(__file__).resolve().parent / "monitoring.db"


def init_monitor_db():
    conn = sqlite3.connect(str(MONITOR_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            sources TEXT,
            response_time REAL,
            user_id INTEGER,
            session_id TEXT,
            user_feedback TEXT,
            feedback_time TEXT,
            category TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_log_id INTEGER,
            source_name TEXT,
            rank INTEGER,
            score REAL,
            chunk_text TEXT,
            FOREIGN KEY (query_log_id) REFERENCES query_logs(id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_logs_created
        ON query_logs(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_logs_feedback
        ON query_logs(user_feedback)
    """)
    conn.commit()
    conn.close()


def log_query(
    question: str,
    answer: str = "",
    sources: List[str] = None,
    response_time: float = 0.0,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    retrieval_docs: List[Dict] = None,
) -> int:
    init_monitor_db()
    conn = sqlite3.connect(str(MONITOR_DB_PATH))
    cursor = conn.cursor()

    sources_json = json.dumps(sources or [], ensure_ascii=False)
    cursor.execute(
        """INSERT INTO query_logs (question, answer, sources, response_time, user_id, session_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (question, answer, sources_json, response_time, user_id, session_id),
    )
    query_log_id = cursor.lastrowid

    if retrieval_docs:
        for rank, doc in enumerate(retrieval_docs, 1):
            cursor.execute(
                """INSERT INTO retrieval_logs (query_log_id, source_name, rank, score, chunk_text)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    query_log_id,
                    doc.get("source", "未知"),
                    rank,
                    doc.get("score", 0.0),
                    doc.get("chunk_text", "")[:500],
                ),
            )

    conn.commit()
    conn.close()
    return query_log_id


def record_feedback(
    query_log_id: int,
    feedback: str,
):
    conn = sqlite3.connect(str(MONITOR_DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE query_logs
           SET user_feedback = ?, feedback_time = datetime('now', 'localtime')
           WHERE id = ?""",
        (feedback, query_log_id),
    )
    conn.commit()
    conn.close()


def get_recent_queries(days: int = 7, limit: int = 100) -> List[Dict]:
    init_monitor_db()
    conn = sqlite3.connect(str(MONITOR_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """SELECT * FROM query_logs
           WHERE created_at >= ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (cutoff, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_statistics(days: int = 7) -> Dict:
    init_monitor_db()
    conn = sqlite3.connect(str(MONITOR_DB_PATH))
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "SELECT COUNT(*) FROM query_logs WHERE created_at >= ?",
        (cutoff,),
    )
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT AVG(response_time) FROM query_logs WHERE created_at >= ? AND response_time > 0",
        (cutoff,),
    )
    avg_time = cursor.fetchone()[0]

    cursor.execute(
        "SELECT user_feedback, COUNT(*) FROM query_logs WHERE created_at >= ? AND user_feedback IS NOT NULL GROUP BY user_feedback",
        (cutoff,),
    )
    feedback_rows = cursor.fetchall()

    cursor.execute(
        """SELECT date(created_at) as day, COUNT(*) as cnt
           FROM query_logs
           WHERE created_at >= ?
           GROUP BY day
           ORDER BY day""",
        (cutoff,),
    )
    daily = cursor.fetchall()

    cursor.execute(
        "SELECT question, COUNT(*) as cnt FROM query_logs WHERE created_at >= ? GROUP BY question ORDER BY cnt DESC LIMIT 10",
        (cutoff,),
    )
    top_queries = cursor.fetchall()

    conn.close()

    feedback_map = {}
    for fb, cnt in feedback_rows:
        feedback_map[fb] = cnt

    return {
        "total_queries": total,
        "avg_response_time": round(avg_time, 2) if avg_time else 0,
        "feedback": feedback_map,
        "daily_trend": [{"date": d, "count": c} for d, c in daily],
        "top_queries": [{"question": q[:80], "count": c} for q, c in top_queries],
    }


def print_report(days: int = 7):
    stats = get_statistics(days)

    print("\n" + "=" * 80)
    print("  系统监控报告")
    print("=" * 80)
    print(f"\n统计周期: 最近 {days} 天")
    print(f"总查询数: {stats['total_queries']}")
    print(f"平均响应时间: {stats['avg_response_time']:.2f} 秒")

    feedback = stats["feedback"]
    if feedback:
        satisfied = feedback.get("satisfied", 0)
        dissatisfied = feedback.get("dissatisfied", 0)
        total_fb = sum(feedback.values())
        print(f"\n用户反馈:")
        print(f"  满意: {satisfied}")
        print(f"  不满意: {dissatisfied}")
        if total_fb > 0:
            print(f"  满意度: {satisfied / total_fb:.2%}")

    print(f"\n每日查询趋势:")
    for day_data in stats["daily_trend"]:
        bar = "█" * min(day_data["count"], 50)
        print(f"  {day_data['date']}  {day_data['count']:>4d}  {bar}")

    if stats["top_queries"]:
        print(f"\n高频查询 Top 10:")
        for i, q in enumerate(stats["top_queries"], 1):
            print(f"  {i:2d}. [{q['count']}次] {q['question']}")

    print("=" * 80)


def export_queries(output_path: str, days: int = 30):
    queries = get_recent_queries(days=days, limit=10000)

    export_data = []
    for q in queries:
        export_data.append({
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"][:200] if q["answer"] else "",
            "sources": json.loads(q["sources"]) if q["sources"] else [],
            "response_time": q["response_time"],
            "feedback": q.get("user_feedback", ""),
            "created_at": q["created_at"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已导出 {len(export_data)} 条查询到 {output_path}")


def sample_evaluation(days: int = 7, sample_size: int = 20):
    print(f"\n正在从最近 {days} 天的查询中抽样评估...")

    queries = get_recent_queries(days=days, limit=1000)
    if not queries:
        print("❌ 没有查询记录")
        return

    import random
    sample = random.sample(queries, min(sample_size, len(queries)))

    has_feedback = [q for q in sample if q.get("user_feedback")]
    no_feedback = [q for q in sample if not q.get("user_feedback")]

    print(f"\n📊 抽样评估结果:")
    print(f"  样本数: {len(sample)}")
    print(f"  有反馈: {len(has_feedback)}")
    print(f"  无反馈: {len(no_feedback)}")

    if has_feedback:
        satisfied = sum(1 for q in has_feedback if q["user_feedback"] == "satisfied")
        print(f"  满意率: {satisfied / len(has_feedback):.2%}")

    response_times = [q["response_time"] for q in sample if q["response_time"] > 0]
    if response_times:
        print(f"  平均响应时间: {statistics.mean(response_times):.2f}s")
        print(f"  P50: {statistics.median(response_times):.2f}s")
        if len(response_times) >= 4:
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            print(f"  P95: {sorted_times[p95_idx]:.2f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="系统监控工具")
    parser.add_argument(
        "--report",
        action="store_true",
        help="查看监控报告",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="统计天数 (默认: 7)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="导出查询数据到JSON文件",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="抽样评估",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="抽样数量 (默认: 20)",
    )
    args = parser.parse_args()

    if args.export:
        export_queries(args.export, days=args.days)
    elif args.sample:
        sample_evaluation(days=args.days, sample_size=args.sample_size)
    elif args.report:
        print_report(days=args.days)
    else:
        print_report(days=args.days)


if __name__ == "__main__":
    main()