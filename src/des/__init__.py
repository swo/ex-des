from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

import altair as alt
import numpy as np
import numpy.random
import polars as pl
import scipy.integrate


@dataclass
class Event:
    time: float
    fun: Callable
    kwargs: dict | None = None


@dataclass
class State:
    compartments: list[Literal["s", "i", "r"]]
    n: int
    beta: float
    gamma: float
    rng: numpy.random.Generator


class Simulation:
    def __init__(
        self, n: int, i0: int, R0: float, gamma: float, end_time: float, seed=None
    ):
        self.state = State(
            compartments=["s"] * n,
            n=n,
            beta=R0 * gamma,
            gamma=gamma,
            rng=numpy.random.default_rng(seed),
        )
        self.time = 0
        self.events: list[Event] = []
        # the statistics ("info") will be a timeseries of counts of each compartment
        self.info = []
        self.end_time = end_time

        # initialize the simulation
        # perform the initial infections
        for j in range(i0):
            self.infect(j)

        # schedule the first contact
        self.schedule_contact()

    def run(self) -> Self:
        while self.events:
            self.events.sort(key=lambda event: event.time)
            next_event = self.events.pop(0)
            if next_event.time > self.end_time:
                break
            else:
                self.resolve(next_event)

        self.time = self.end_time
        return self

    def resolve(self, event: Event) -> None:
        if event.time < self.time:
            raise RuntimeError

        self.time = event.time
        event.fun(**(event.kwargs or {}))

    def schedule(self, event: Event) -> None:
        self.events.append(event)

    def schedule_contact(self) -> None:
        contact_rate = self.state.beta * self.state.n
        delay = self.state.rng.exponential(scale=1.0 / contact_rate)
        self.schedule(Event(time=self.time + delay, fun=self.contact))

    def contact(self):
        infector, infectee = self.state.rng.choice(
            range(self.state.n), size=2, replace=False
        )

        # note that we're interested in I->S, not also S<-I, because that's
        # equivalent to doubling the rate of contacts
        if (
            self.state.compartments[infector] == "i"
            and self.state.compartments[infectee] == "s"
        ):
            self.infect(infectee)

        self.schedule_contact()

    def infect(self, person: int) -> None:
        if self.state.compartments[person] != "s":
            raise RuntimeError

        self.state.compartments[person] = "i"
        self.schedule_recovery(person=person)
        self.report_timeseries()

    def schedule_recovery(self, person: int) -> None:
        delay = self.state.rng.exponential(scale=1.0 / self.state.gamma)
        self.schedule(
            Event(time=self.time + delay, fun=self.recover, kwargs={"person": person})
        )

    def recover(self, person: int) -> None:
        if self.state.compartments[person] != "i":
            raise RuntimeError

        self.state.compartments[person] = "r"
        self.report_timeseries()

    def report_timeseries(self) -> None:
        counts = [self.state.compartments.count(c) for c in ["s", "i", "r"]]
        self.info.append((self.time, *counts))

    def to_df(self) -> pl.DataFrame:
        return pl.from_records(
            self.info, orient="row", schema=["t", "s", "i", "r"]
        ).unpivot(index="t")


def ode(n: float, i0: float, R0: float, gamma: float, end_time: float) -> pl.DataFrame:
    """ODE solution of the SIR model"""
    beta = R0 * gamma

    def rates(_, y):
        s, i, _ = y
        ds = -beta * s * i / n
        dr = gamma * i
        di = -ds - dr
        return (ds, di, dr)

    res = scipy.integrate.solve_ivp(
        rates,
        t_span=(0.0, end_time),
        y0=(n - i0, i0, 0.0),
        t_eval=np.linspace(0.0, end_time, num=101),
        dense_output=True,
    )

    return (
        pl.from_numpy(res.y.T, schema=["s", "i", "r"])
        .with_columns(t=res.t)
        .unpivot(index="t")
    )


def main():
    n = 1000
    i0 = 10
    R0 = 1.5
    gamma = 0.25
    end_time = 100.0

    sim_df = Simulation(n=n, i0=i0, R0=R0, gamma=gamma, end_time=end_time).run().to_df()
    ode_df = ode(n=n, i0=i0, R0=R0, gamma=gamma, end_time=end_time)

    df = pl.concat(
        [
            sim_df.with_columns(pl.col("value").cast(pl.Float64), source=pl.lit("sim")),
            ode_df.with_columns(source=pl.lit("ode")),
        ]
    )

    chart = (
        alt.Chart(df)
        .encode(alt.X("t"), alt.Facet("variable"), alt.Y("value"), alt.Color("source"))
        .mark_line()
        .resolve_scale(y="independent")
    )
    chart.save("chart.png")
