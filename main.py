"""ЛР1: вещественный генетический алгоритм для 7-мерной функции Розенброка.

Зависимости: numpy, matplotlib. Все эволюционные операторы реализованы здесь.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

DIM = 7  # N=15, Q=0 (группа 507), Y=26 => V=2; d=5+(2 mod 6)
LOWER, UPPER = -5.0, 10.0
OPTIMUM = np.ones(DIM, dtype=float)


def rosenbrock(points: np.ndarray) -> np.ndarray:
    """f(x) = sum_{i=1}^{d-1} [100(x_{i+1}-x_i^2)^2+(x_i-1)^2].

    Accepts a single vector (d,) or a batch (n,d); returns vector of scores.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"Ожидается массив формы (n, {DIM})")
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)


@dataclass(frozen=True)
class Config:
    population: int = 100
    evaluations: int = 40_000
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2  # вероятность для КАЖДОЙ координаты
    sigma_start: float = 0.7
    sigma_end: float = 0.025
    blx_alpha: float = 0.5
    tournament_size: int = 3
    elite: int = 2

    def validate(self) -> None:
        if self.population < 30 or self.evaluations < self.population:
            raise ValueError("Популяция >= 30, бюджет >= размера популяции")
        if self.evaluations % self.population:
            raise ValueError("Бюджет должен быть кратен размеру популяции")
        if not 0 <= self.mutation_probability <= 1 or not 0 <= self.crossover_probability <= 1:
            raise ValueError("Вероятности должны быть от 0 до 1")
        if self.sigma_start <= 0 or self.sigma_end <= 0:
            raise ValueError("Сигмы должны быть положительными")
        if self.elite < 1 or self.elite >= self.population:
            raise ValueError("Недопустимое число элит")
        if self.tournament_size < 2:
            raise ValueError("Размер турнира должен быть >= 2")
        if self.blx_alpha < 0:
            raise ValueError("BLX-alpha параметр должен быть >= 0")


def tournament(fitness: np.ndarray, rng: np.random.Generator, size: int,
               count: int = 1) -> int | np.ndarray:
    """Турнирный отбор одной или нескольких особей (минимизируем fitness).

    При count > 1 выбираются независимые победители, по одному на строку.
    """
    if size < 2 or count < 1:
        raise ValueError("Размер турнира >= 2, число отборов >= 1")
    indices = rng.integers(len(fitness), size=(count, size))
    winners = indices[np.arange(count), np.argmin(fitness[indices], axis=1)]
    return int(winners[0]) if count == 1 else winners


def blx_alpha_crossover(a: np.ndarray, b: np.ndarray, alpha: float,
                        rng: np.random.Generator) -> np.ndarray:
    """BLX-alpha: покоординатная равномерная выборка из расширенного интервала."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    span = hi - lo
    return rng.uniform(lo - alpha * span, hi + alpha * span)


def mutate(child: np.ndarray, probability: float, sigma: float,
           rng: np.random.Generator) -> np.ndarray:
    """Гауссовская мутация вектора (d,) или пакета векторов (n, d).

    У каждой координаты собственный бросок вероятности; результат проецируется
    в допустимый гиперкуб. Пакетный режим применён и в основном цикле ГА.
    """
    result = np.asarray(child, dtype=float).copy()
    if result.ndim not in (1, 2) or result.shape[-1] != DIM:
        raise ValueError(f"Ожидался вектор (..., {DIM}) с 1 или 2 измерениями")
    mask = rng.random(result.shape) < probability
    if result.ndim == 1:
        # Сохраняем экономный режим для одиночного потомка.
        result[mask] += rng.normal(0, sigma, size=int(mask.sum()))
    else:
        # Генерируем нормальные шумы одинаковой формы для всех потомков.
        result += mask * rng.normal(0.0, sigma, size=result.shape)
    return np.clip(result, LOWER, UPPER)


def ga(config: Config, seed: int) -> dict:
    """Минимизирует f. Возвращает историю лучших и точный расход фитнес-вызовов."""
    config.validate()
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LOWER, UPPER, size=(config.population, DIM))
    fitness = rosenbrock(pop)
    count = config.population
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    history = [best_f]
    generations = config.evaluations // config.population

    for generation in range(1, generations):
        # Сохраняем лучшие решения (элитизм).
        elite_indices = np.argsort(fitness)[:config.elite]
        # Уменьшение масштаба мутации заранее задаётся расписанием, без подбора по тесту.
        fraction = generation / (generations - 1)
        sigma = config.sigma_start * (config.sigma_end / config.sigma_start) ** fraction
        n_children = config.population - config.elite
        # Те же проверяемые операторы применяются и при пакетной обработке потомков.
        winners_a = tournament(fitness, rng, config.tournament_size, count=n_children)
        winners_b = tournament(fitness, rng, config.tournament_size, count=n_children)
        first, second = pop[winners_a], pop[winners_b]
        crossed = blx_alpha_crossover(first, second, config.blx_alpha, rng)
        cross_mask = rng.random(n_children) < config.crossover_probability
        children = np.where(cross_mask[:, None], crossed, first).copy()
        children = mutate(children, config.mutation_probability, sigma, rng)
        pop = np.vstack([pop[elite_indices], children])
        # Переоценка элиты учитывается в общем бюджете, как и остальные особи.
        fitness = rosenbrock(pop)
        count += config.population
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_f:
            best_f = float(fitness[idx])
            best_x = pop[idx].copy()
        history.append(best_f)
    assert count == config.evaluations
    return {"seed": seed, "best_f": best_f, "best_x": best_x.tolist(),
            "history": history, "evaluations": count}


def random_search(config: Config, seed: int) -> dict:
    """Честный baseline: равномерные независимые векторы, тот же бюджет."""
    rng = np.random.default_rng(seed)
    best_f = float("inf")
    best_x = None
    history = []
    for _ in range(config.evaluations // config.population):
        samples = rng.uniform(LOWER, UPPER, size=(config.population, DIM))
        scores = rosenbrock(samples)
        idx = int(np.argmin(scores))
        if scores[idx] < best_f:
            best_f, best_x = float(scores[idx]), samples[idx].copy()
        history.append(best_f)
    return {"seed": seed, "best_f": best_f, "best_x": best_x.tolist(),
            "history": history, "evaluations": config.evaluations}


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"best": float(np.min(a)), "mean": float(np.mean(a)),
            "median": float(np.median(a)), "std": float(np.std(a, ddof=1)),
            "worst": float(np.max(a))}


def paired_t_comparison(values_a: list[float], values_b: list[float]) -> dict:
    """Двусторонний парный t-тест по результатам для одинаковых seed.

    Тест — описательное сравнение двух заранее заданных конфигураций;
    нормальность распределения парных разностей не проверяется автоматически.
    """
    from scipy.stats import ttest_rel

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2:
        raise ValueError("Для парного t-теста нужны две серии равной длины (>= 2)")
    statistic, p_value = ttest_rel(a, b)
    return {
        "method": "paired_two_sided_student_t_test",
        "difference": "GA_p0.20 minus GA_p0.40",
        "mean_difference": float(np.mean(a - b)),
        "t_statistic": float(statistic),
        "degrees_of_freedom": int(len(a) - 1),
        "p_value": float(p_value),
        "alpha": 0.05,
        "significant_at_0_05": bool(p_value < 0.05),
    }


def run_experiments(out: Path, runs: int, base_seed: int, config: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    configs = {
        "GA_p0.20": config,
        "GA_p0.40": replace(config, mutation_probability=0.4),
    }
    with (out / "experiment_config.json").open("w", encoding="utf-8") as fp:
        json.dump({"variant": 2, "N": 15, "Q": 0, "Y": 26, "dimension": DIM,
                   "domain": [LOWER, UPPER], "runs": runs, "base_seed": base_seed,
                   "configs": {k: vars(v) for k, v in configs.items()},
                   "random_baseline": "uniform sampling same evaluation budget"},
                  fp, ensure_ascii=False, indent=2)
    all_data = {}
    # Парные seed для сопоставимости условий. Разные алгоритмы используют
    # собственные потоки ГПСЧ при одинаковом начальном seed.
    for name, cfg in [*configs.items(), ("Random", config)]:
        results = []
        for i in range(runs):
            seed = base_seed + i
            result = random_search(cfg, seed) if name == "Random" else ga(cfg, seed)
            results.append(result)
        all_data[name] = results
        print(f"{name}: {summary([r['best_f'] for r in results])}")

    with (out / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "seed", "best_f", "evaluations", *[f"x{i}" for i in range(1, DIM + 1)]])
        for name, results in all_data.items():
            for r in results:
                writer.writerow([name, r["seed"], f"{r['best_f']:.12g}", r["evaluations"],
                                 *[f"{v:.12g}" for v in r["best_x"]]])

    with (out / "convergence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "generation", "evaluation", "min", "mean", "max"])
        for name, results in all_data.items():
            matrix = np.asarray([r["history"] for r in results])
            for j in range(matrix.shape[1]):
                writer.writerow([name, j + 1, (j + 1) * config.population,
                                 float(np.min(matrix[:, j])), float(np.mean(matrix[:, j])),
                                 float(np.max(matrix[:, j]))])

    stats = {name: summary([r["best_f"] for r in results]) for name, results in all_data.items()}
    comparison = paired_t_comparison([r["best_f"] for r in all_data["GA_p0.20"]],
                                     [r["best_f"] for r in all_data["GA_p0.40"]])
    best_method = min(all_data["GA_p0.20"], key=lambda r: r["best_f"])
    with (out / "stats.json").open("w", encoding="utf-8") as f:
        json.dump({"statistics": stats, "parameter_comparison": comparison,
                   "best_GA_p0.20": best_method,
                   "theoretical_optimum": {"x": OPTIMUM.tolist(), "f": 0.0}},
                  f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, results in all_data.items():
        arr = np.asarray([r["history"] for r in results])
        xs = np.arange(1, arr.shape[1] + 1)
        ax.plot(xs, np.maximum(np.mean(arr, axis=0), 1e-12), label=f"{name}: mean")
        ax.fill_between(xs, np.maximum(np.min(arr, axis=0), 1e-12),
                        np.maximum(np.max(arr, axis=0), 1e-12), alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Generation (evaluated population)")
    ax.set_ylabel("Best-so-far f(x), log scale")
    ax.set_title(f"Rosenbrock, d={DIM}, {runs} independent runs (shading: min–max)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(all_data)
    vectors = [[r["best_f"] for r in all_data[name]] for name in labels]
    ax.boxplot(vectors, tick_labels=labels, showmeans=True)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("Final best f(x), symlog scale")
    ax.set_title(f"Final results of {runs} runs per method")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "boxplot.png", dpi=180)
    plt.close(fig)

    def fmt(x: float) -> str:
        return f"{x:.6g}"
    lines = [
        "# Отчёт по лабораторной работе № 1",
        "", "**Дисциплина:** Эволюционное программирование и генетические алгоритмы  ",
        "**Тема:** Многомерная оптимизация сложной функции генетическим алгоритмом  ",
        "**Группа:** Б24-507  ", "**Номер по списку:** 15  ",
        "**Вариант:** 2 (функция Розенброка)", "",
        "## 1. Постановка задачи и определение варианта", "",
        "Согласно методическим указаниям, N=15, Q=0 для группы 507, Y=26 для 2026 года.",
        "По формуле V = 1 + ((17N + 7Q + Y) mod 20) получаем V = 1 + (281 mod 20) = 2.",
        "Размерность d = 5 + (V mod 6) = 7.", "",
        "Нужно минимизировать функцию Розенброка", "",
        "$$f(x)=\\sum_{i=1}^{6}\\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\\right],$$", "",
        "где $x\\in[-5,10]^7$. Теоретический глобальный минимум равен 0 при $x=(1,1,1,1,1,1,1)$.",
        "Все слагаемые функции неотрицательны, а при x=(1,...,1) обращаются в нуль.", "",
        "## 2. Кодирование, ограничения, фитнес", "",
        "Особь (генотип) представлена вещественным вектором длины 7; фенотип совпадает с генотипом, декодирование не требуется.",
        "Допустимые решения лежат в гиперкубе [-5,10]^7. Начальная популяция генерируется равномерно в гиперкубе.",
        "Фитнесом служит исходная целевая функция (меньше — лучше). После изменения координат применяется отсечение (`clip`) к допустимым границам.", "",
        "## 3. Генетический алгоритм", "",
        "Родители выбираются турниром из 3 случайных особей (побеждает меньший фитнес).",
        "Рекомбинация BLX-α выполняется с вероятностью 0.9: для каждой координаты потомок выбирается равномерно из",
        "$[\\min(a_i,b_i)-\\alpha|a_i-b_i|,\\max(a_i,b_i)+\\alpha|a_i-b_i|]$, где α=0.5.",
        "Каждая координата потомка мутирует независимо с вероятностью p: к ней прибавляется шум N(0,σ²).",
        "Для перехода от глобального к точному поиску σ геометрически уменьшается с 0.7 до 0.025 по поколениям.",
        "После кроссовера и мутации потомок проецируется на границы области. Две лучшие особи каждого поколения переносятся без изменений (элитизм).",
        "Турнирная селекция создаёт умеренное давление отбора: предпочитает перспективных родителей, но не исключает разнообразие.",
        "Элитизм удерживает лучшие найденные решения от потери при случайной мутации и рекомбинации;",
        "это важно при поиске вдоль узкой изогнутой долины функции Розенброка.",
        "Остановка происходит строго при фиксированном числе оценок функции, включая начальную популяцию.", "",
        "### Блок-схема", "", "```mermaid",
        "flowchart TD", " A[Установить seed и параметры] --> B[Случайная начальная популяция]",
        " B --> C[Вычислить фитнес]", " C --> D{Бюджет исчерпан?}",
        " D -- Да --> E[Сохранить лучший результат]", " D -- Нет --> F[Сохранить элиту]",
        " F --> G[Турнирная селекция]", " G --> H[BLX-alpha кроссовер]",
        " H --> I[Гауссовская мутация и clip]", " I --> J[Новое поколение и фитнес]",
        " J --> D", "```", "",
        "![Блок-схема генетического алгоритма](results/flowchart.png)", "",
        "*Рис. 1. Блок-схема эволюционного цикла (приложена также как PNG для просмотра без Mermaid).*", "",
        "## 4. Экспериментальный протокол", "",
        f"Проведено {runs} независимых запусков каждого метода с seed от {base_seed} до {base_seed + runs - 1}.",
        f"В каждом запуске бюджет ровно {config.evaluations} оценок целевой функции; размер популяции {config.population},",
        f"число оценённых поколений {config.evaluations // config.population}, элитизм {config.elite}, BLX-α {config.blx_alpha}.",
        "Единственный изменяемый фактор в двух конфигурациях ГА — покоординатная вероятность мутации: p=0.20 и p=0.40.",
        "Для baseline использован независимый равномерный случайный поиск по [-5,10]^7 с тем же бюджетом.",
        "Все итоги сохранены в `results/runs.csv`, траектории по поколениям (с дополнительным столбцом оценок функции) в `results/convergence.csv`, параметры в `results/experiment_config.json`.", "",
        "## 5. Экспериментальные результаты", "",
        "| Алгоритм | Лучший | Средний | Медиана | Std (выборочное) | Худший |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, st in stats.items():
        lines.append(f"| {name} | {fmt(st['best'])} | {fmt(st['mean'])} | {fmt(st['median'])} | {fmt(st['std'])} | {fmt(st['worst'])} |")
    lines += ["", "![График сходимости](results/convergence.png)", "",
              f"*Рис. 2. Лучшее найденное значение по поколениям: средняя траектория и диапазон min–max по {runs} запускам каждого метода. В каждой генерации оценивается {config.population} особей.*",
              "", "![Диаграмма итоговых результатов](results/boxplot.png)", "",
              "*Рис. 3. Распределение лучшего значения функции по итогам каждого запуска.*", "",
              "Лучшее решение базовой конфигурации (p=0.20):", "",
              f"- Seed: `{best_method['seed']}`", f"- f(x): `{fmt(best_method['best_f'])}`",
              f"- x: `{[round(v, 7) for v in best_method['best_x']]}`", "",
              "## 6. Анализ и выводы", ""]
    main_mean = stats["GA_p0.20"]["mean"]
    alt_mean = stats["GA_p0.40"]["mean"]
    baseline_mean = stats["Random"]["mean"]
    advantage = "ниже" if main_mean < alt_mean else "выше"
    lines += [
        f"Среднее итоговое значение f у ГА с p=0.20 равно {fmt(main_mean)}, что {advantage} среднего для p=0.40 ({fmt(alt_mean)}).",
        "Другие параметры ГА и бюджет совпадают, поэтому сравнение изолирует изменяемую вероятность мутации.",
        f"Парный двусторонний t-тест по одинаковым seed: средняя разность f(p=0.20) − f(p=0.40) "
        f"равна {fmt(comparison['mean_difference'])}; t({comparison['degrees_of_freedom']}) = "
        f"{fmt(comparison['t_statistic'])}, p = {comparison['p_value']:.4f}.",
        ("Различие статистически значимо на уровне 0.05." if comparison['significant_at_0_05']
         else "На уровне значимости 0.05 статистически убедимого различия не обнаружено; "
              "не следует утверждать, что p=0.40 гарантированно лучше p=0.20."),
        "Парный t-тест предполагает приблизительную нормальность разностей; при 20 запусках вывод следует "
        "интерпретировать осторожно, без обещаний для других наборов параметров или функций.",
        f"Средний результат случайного поиска равен {fmt(baseline_mean)}. " +
        ("ГА с p=0.20 в среднем эффективнее baseline при равном бюджете. " if main_mean < baseline_mean else
         "В этих условиях ГА с p=0.20 не превосходит baseline в среднем. ") +
        "Это эмпирическое сравнение, а не гарантия превосходства для любого seed.",
        "Функция Розенброка затрудняет оптимизацию узкой изогнутой долиной: координаты сильно связаны, и",
        "движение в сторону оптимума требует согласованного изменения нескольких координат. "
        "Равномерный случайный поиск редко попадает близко к такой долине в семи измерениях, "
        "а локальный поиск с неудачным шагом способен долго двигаться вдоль неё.",
        "Метод BLX-α позволяет комбинировать координаты родителей и исследовать области около них; "
        "гауссовская мутация поддерживает разнообразие и делает точное уточнение при уменьшении σ.",
        "Качество следует оценивать относительно известного теоретического минимума f*=0. "
        "Чем ближе экспериментальный f к нулю и чем меньше разброс между запусками, тем надёжнее результат.",
        "Ограничения: фиксированный бюджет, чувствительность к параметрам, отсутствие гарантии глобального "
        "оптимума и проверка лишь на одной тестовой функции.", "",
        "## 7. Воспроизводимость", "",
        "`python main.py --runs 20 --seed 2026 --out results`", "",
        "Запуск полностью пересоздаёт таблицы и графики. Для проверки корректности: `python -m unittest discover -s tests`.", "",
        "## Источник", "", "Душкин Р. В. «Эволюционное программирование и генетические алгоритмы», комплекс лабораторных работ, ЛР № 1, с. 4–8.",
    ]
    (out.parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР1, вариант 2, Rosenbrock d=7")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--evaluations", type=int, default=40000)
    parser.add_argument("--population", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 20:
        raise ValueError("По заданию требуется не менее 20 запусков")
    run_experiments(args.out, args.runs, args.seed,
                    Config(evaluations=args.evaluations, population=args.population))


if __name__ == "__main__":
    main()
