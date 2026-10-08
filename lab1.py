from enum import Enum
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns


class Dist:
    def __init__(self, name, distribution):
        self.name = name
        self.distribution = distribution


class Rvs(Enum):
    NORMAL = Dist("нормального", stats.norm(loc=0, scale=1))


# Функция для построения графиков функций распределения
def plot_distribution(dist: Dist):
    rv = dist.distribution
    name = dist.name

    sample = rv.rvs(10000, random_state=1)  # выборка

    # 1: картинка плотности распределения с точками -1, 1 и 2
    x = np.linspace(-3, 3, 100)
    pdf = rv.pdf(x)

    plt.title(f"График плотности {name} распределения c точками в x = -1, 1 и 2")
    plt.plot(x, pdf)
    plt.scatter([-1, 1, 2], [rv.pdf(-1), rv.pdf(1), rv.pdf(2)], color="blue")

    plt.show()

    # 2: график значения функции распределения с точками в x = 0, 1 и 2

    plt.title(f"График плотности {name} распределения c точками в x = 0, 1 и 2")
    plt.plot(x, pdf)
    plt.ylabel("$f(x)$")
    plt.xlabel("$x$")
    plt.scatter([0, 1, 2], [rv.pdf(0), rv.pdf(1), rv.pdf(2)], color="blue")

    plt.show()

    print(f"Значение функции {name} распределения в точке 1: {rv.cdf(1)}")

    # 3: вероятность, что X <= 1 на графике плотности распределения
    x = np.linspace(-3, 3, 100)
    pdf = rv.pdf(x)

    plt.title(f"Вероятность, что $X$ <= 1 для {name} распределения")

    plt.plot(x, pdf)
    plt.ylabel("$f(x)$")
    plt.xlabel("$x$")

    plt.scatter([1], [rv.pdf(1)], color="blue")

    xq = np.linspace(-3, 1, 100)
    yq = rv.pdf(xq)
    plt.fill_between(xq, 0, yq, color="blue", alpha=0.2)

    plt.axvline(1, color="blue", linestyle="--", lw=2)

    plt.show()

    # 4: построение графика функции распределения
    x = np.linspace(-3, 3, 100)
    cdf = rv.cdf(x)

    plt.title(f"График функции {name} распределения")
    plt.plot(x, cdf)
    plt.ylabel("$F(x)$")
    plt.xlabel("$x$")

    plt.scatter([1], [rv.cdf(1)], color="blue")

    plt.show()

    # 5. График вероятности попасть в конкретный отрезок на плотности распределения
    x = np.linspace(-5, 5, 100)
    pdf = rv.pdf(x)

    plt.title(f"Вероятность попасть в отрезок на плотности {name} распределения")
    plt.plot(x, pdf)
    plt.ylabel("$f(x)$")
    plt.xlabel("$x$")

    # На ней же нарисуем f(1)
    plt.scatter([1, 3], [rv.pdf(1), rv.pdf(3)], color="blue")

    # на ту же картинку добавили новую часть, штриховку
    xq = np.linspace(1, 3)
    yq = rv.pdf(xq)
    plt.fill_between(xq, 0, yq, color="blue", alpha=0.2)

    plt.axvline(1, color="blue", linestyle="--", lw=2)
    plt.axvline(3, color="blue", linestyle="--", lw=2)

    plt.show()

    # 6. Изображаем квантили
    x = np.linspace(-3, 3, 100)
    pdf = rv.pdf(x)

    quantile = 0.05

    np_q = np.quantile(sample, quantile)

    plt.title(f"График вероятностей для квантиля {quantile}")
    plt.plot(x, pdf)
    plt.ylabel("$f(x)$")
    plt.xlabel("$x$")

    xq = np.linspace(-3, np_q)
    yq = rv.pdf(xq)
    plt.fill_between(xq, 0, yq, color="blue", alpha=0.2)

    plt.axvline(np_q, color="blue", linestyle="--", lw=2)

    y_max = plt.ylim()[1]
    plt.text(np_q + 0.1, 0.8 * y_max, round(np_q, 2), color="blue", fontsize=16)
    plt.show()

    print(f"Выборочное среднее: {np.mean(sample)}")

    x = np.linspace(-3, 3, 100)
    pdf = rv.pdf(x)

    plt.plot(x, pdf, lw=3)
    plt.title(f"Гистограмма выборки и плотность {name} распределения")

    # гистограмма, параметр density отнормировал её.
    plt.hist(sample, bins=100, density=True)

    plt.ylabel("$f(x)$")
    plt.xlabel("$x$")
    plt.show()
    
    


if __name__ == "__main__":
    rasp = Rvs.NORMAL.value
    plot_distribution(rasp)
