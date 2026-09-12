import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self


@dataclass
class Event:
    time: float
    fun: Callable
    kwargs: dict


class Simulation:
    def __init__(self, end_time: float):
        self.state = {}
        self.time = 0
        self.events: list[Event] = []
        self.info = {}
        self.end_time = end_time

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
        self.time = event.time
        event.fun(**event.kwargs)

    def schedule(self, event: Event) -> None:
        self.events.append(event)


class SirPair(Simulation):
    def __init__(
        self,
        n: int,
        i0: int,
        R0: float,
        gamma: float,
        end_time: float,
        seed=None,
    ):
        """SIR simulation, where the loop is over contacts between all pairs of people"""
        super().__init__(end_time=end_time)
        random.seed(seed)
        self.state["compartment"] = ["s"] * n
        self.state["n"] = n
        self.state["contact_rate"] = R0 * gamma * n
        self.state["gamma"] = gamma
        self.info["timeseries"] = []

        # perform the initial infections
        for j in range(i0):
            self.infect(j)

        # schedule the first contact
        self.schedule_contact()

    def schedule_contact(self) -> None:
        delay = random.expovariate(lambd=self.state["contact_rate"])
        self.schedule(Event(time=self.time + delay, fun=self.contact, kwargs={}))

    def contact(self):
        person1, person2 = random.sample(range(self.state["n"]), 2)
        comp1 = self.state["compartment"][person1]
        comp2 = self.state["compartment"][person2]
        if comp1 == "s" and comp2 == "i":
            self.infect(person1)
        elif comp1 == "i" and comp2 == "s":
            self.infect(person2)
        else:
            pass

        self.schedule_contact()

    def infect(self, person: int) -> None:
        if self.state["compartment"][person] != "s":
            raise RuntimeError

        self.state["compartment"][person] = "i"
        self.report_timeseries()

        self.schedule_recovery(person=person)

    def schedule_recovery(self, person: int) -> None:
        delay = random.expovariate(lambd=self.state["gamma"])
        self.schedule(
            Event(time=self.time + delay, fun=self.recover, kwargs={"person": person})
        )

    def recover(self, person: int) -> None:
        if self.state["compartment"][person] != "i":
            raise RuntimeError

        self.state["compartment"][person] = "r"
        self.report_timeseries()

    def report_timeseries(self) -> None:
        counts = [self.state["compartment"].count(c) for c in ["s", "i", "r"]]
        self.info["timeseries"].append((self.time, *counts))


def main():
    n = 1000
    i0 = 10
    R0 = 1.25
    gamma = 0.25
    end_time = 100.0
    sim = SirPair(n=n, i0=i0, R0=R0, gamma=gamma, end_time=end_time).run()

    import altair as alt
    import polars as pl
    import scipy.integrate

    sim_df = pl.from_records(
        sim.info["timeseries"], orient="row", schema=["t", "s", "i", "r"]
    ).unpivot(index="t")

    beta = R0 * gamma

    def ode(t, y):
        s, i, _ = y
        ds = -beta * s * i / n
        dr = gamma * i
        di = -ds - dr
        return (ds, di, dr)

    res = scipy.integrate.solve_ivp(
        ode,
        t_span=(0.0, end_time),
        y0=(n - i0, i0, 0.0),
        t_eval=sim_df["t"].unique().sort(),
        dense_output=True,
    )

    ode_df = (
        pl.from_numpy(res.y.T, schema=["s", "i", "r"])
        .with_columns(t=res.t)
        .unpivot(index="t")
    )

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
