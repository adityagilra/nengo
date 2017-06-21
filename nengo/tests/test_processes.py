from __future__ import absolute_import

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.base import Process
from nengo.dists import Distribution, Gaussian
from nengo.exceptions import ValidationError
from nengo.processes import BrownNoise, FilteredNoise, WhiteNoise, WhiteSignal
from nengo.synapses import Lowpass


class DistributionMock(Distribution):
    def __init__(self, retval):
        super(DistributionMock, self).__init__()
        self.retval = retval
        self.sample_calls = []

    def sample(self, n, d, rng=np.random):
        self.sample_calls.append((n, d, rng))
        return np.ones((n, d)) * self.retval


class TimeProcess(Process):
    def make_step(self, shape_in, shape_out, dt, rng):
        size_in = np.prod(shape_in)
        size_out = np.prod(shape_out)
        if size_in == 0:
            return lambda t: [t] * size_out
        else:
            return lambda t, x: [t * np.sum(x)] * size_out


def test_time(RefSimulator):
    trun = 1.
    c = 2.
    process = TimeProcess()

    with nengo.Network() as model:
        u = nengo.Node(process)
        up = nengo.Probe(u)

        q = nengo.Node(c)
        v = nengo.Node(process, size_in=1, size_out=1)
        nengo.Connection(q, v, synapse=None)
        vp = nengo.Probe(v)

    with RefSimulator(model) as sim:
        sim.run(trun)

    nt = len(sim.trange())
    assert process.trange(trun).shape == sim.trange().shape
    assert np.allclose(process.trange(trun), sim.trange())
    assert np.allclose(process.ntrange(nt), sim.trange())
    assert np.allclose(process.run(trun), sim.data[up])
    assert np.allclose(process.apply([c] * nt), sim.data[vp])


def test_whitenoise(rng):
    dist = DistributionMock(42)
    process = WhiteNoise(dist, scale=False)
    samples = process.run_steps(5, d=2, rng=rng)
    assert np.all(samples == 42 * np.ones((5, 2)))
    assert process.run_steps(5, rng=rng).shape == (5, 1)
    assert process.run_steps(1, d=1, rng=rng).shape == (1, 1)
    assert process.run_steps(2, d=3, rng=rng).shape == (2, 3)


def test_brownnoise(Simulator, seed, plt):
    d = 5000
    t = 0.5
    std = 1.5
    process = BrownNoise(dist=Gaussian(0, std))
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    samples = sim.data[up]

    trange = sim.trange()
    expected_std = std * np.sqrt(trange)
    atol = 3.5 * expected_std / np.sqrt(d)

    plt.subplot(2, 1, 1)
    plt.title("Five Brown noise signals")
    plt.plot(trange, samples[:, :5])
    plt.subplot(2, 1, 2)
    plt.ylabel("Standard deviation")
    plt.plot(trange, np.abs(np.std(samples, axis=1)), label="Actual")
    plt.plot(trange, expected_std, label="Expected")
    plt.legend(loc='best')

    assert np.all(np.abs(np.mean(samples, axis=1)) < atol)
    assert np.all(np.abs(np.std(samples, axis=1) - expected_std) < atol)


def psd(values, dt=0.001):
    freq = npext.rfftfreq(values.shape[0], d=dt)
    power = 2. * np.std(np.abs(np.fft.rfft(
        values, axis=0)), axis=1) / np.sqrt(values.shape[0])
    return freq, power


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_whitenoise(Simulator, rms, seed, plt):
    d = 500
    process = WhiteNoise(Gaussian(0., rms), scale=False)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(0.3)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    val_rms = npext.rms(values, axis=0)
    assert np.allclose(val_rms.mean(), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.2)


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_whitesignal_rms(Simulator, rms, seed, plt):
    t = 1.
    d = 500
    process = WhiteSignal(t, high=500, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    assert np.allclose(np.std(values), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.35)


@pytest.mark.parametrize('y0,d', [(0, 1), (-0.3, 3), (0.4, 1)])
def test_whitesignal_y0(Simulator, seed, y0, d):
    t = .1
    process = WhiteSignal(t, high=500, y0=y0)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    values = sim.data[up]
    error = np.min(abs(y0 - values), axis=0)

    assert ((y0 - error <= values[0, :]) & (values[0, :] <= y0 + error)).all()


@pytest.mark.parametrize('high,dt', [(10, 0.01), (5, 0.001), (50, 0.001)])
def test_whitesignal_high_dt(Simulator, high, dt, seed, plt):
    t = 1.
    rms = 0.5
    d = 500
    process = WhiteSignal(t, high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed, dt=dt) as sim:
        sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(len(trange), dt) > high] < rms * 0.5)


@pytest.mark.parametrize('high,dt', [(501, 0.001), (500, 0.002)])
def test_whitesignal_nyquist(Simulator, dt, high, seed):
    # check that high cannot exceed nyquist frequency
    process = WhiteSignal(1.0, high=high)
    with nengo.Network() as model:
        nengo.Node(process, size_out=1)

    with pytest.raises(ValidationError):
        Simulator(model, dt=dt, seed=seed)


def test_whitesignal_continuity(Simulator, seed, plt):
    """Test that WhiteSignal is continuous over multiple periods."""
    t = 1.
    high = 10
    rms = 0.5
    process = WhiteSignal(t, high=high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=1)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(4 * t)
    dt = sim.dt
    x = sim.data[up]

    plt.plot(sim.trange(), x)

    # tolerances approximated from derivatives of sine wave of highest freq
    safety_factor = 2.
    a, f = np.sqrt(2) * rms, (2 * np.pi * high) * dt
    assert abs(np.diff(x, axis=0)).max() <= safety_factor * a * f
    assert abs(np.diff(x, n=2, axis=0)).max() <= safety_factor**2 * a * f**2


def test_sampling_shape():
    process = WhiteSignal(0.1, high=500)
    assert process.run_steps(1).shape == (1, 1)
    assert process.run_steps(5, d=1).shape == (5, 1)
    assert process.run_steps(1, d=2). shape == (1, 2)


def test_reset(Simulator, seed):
    trun = 0.1

    with nengo.Network() as model:
        u = nengo.Node(WhiteNoise(Gaussian(0, 1), scale=False))
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(trun)
        x = np.array(sim.data[up])
        sim.reset()
        sim.run(trun)
        y = np.array(sim.data[up])

    assert x.shape == y.shape
    assert np.allclose(x, y)


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = WhiteNoise(dist=Gaussian(0.3, 0.2))
    b = WhiteNoise(dist=Gaussian(0.3, 0.2))
    c = FilteredNoise(dist=Gaussian(0.3, 0.2), synapse=Lowpass(0.02))

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely

    with pytest.raises(ValueError):
        a.dist = Gaussian(0.3, 0.5)  # test that dist param is frozen
    with pytest.raises(ValueError):
        a.dist.std = 0.4  # test that dist object is frozen


def test_seed(Simulator, seed):
    with nengo.Network() as model:
        a = nengo.Node(WhiteSignal(0.1, high=100, seed=seed))
        b = nengo.Node(WhiteSignal(0.1, high=100, seed=seed+1))
        c = nengo.Node(WhiteSignal(0.1, high=100))
        d = nengo.Node(WhiteNoise(seed=seed))
        e = nengo.Node(WhiteNoise())
        ap = nengo.Probe(a)
        bp = nengo.Probe(b)
        cp = nengo.Probe(c)
        dp = nengo.Probe(d)
        ep = nengo.Probe(e)

    with Simulator(model) as sim1:
        sim1.run(0.1)

    with Simulator(model) as sim2:
        sim2.run(0.1)

    tols = dict(atol=1e-7, rtol=1e-4)
    assert np.allclose(sim1.data[ap], sim2.data[ap], **tols)
    assert np.allclose(sim1.data[bp], sim2.data[bp], **tols)
    assert not np.allclose(sim1.data[cp], sim2.data[cp], **tols)
    assert not np.allclose(sim1.data[ap], sim1.data[bp], **tols)
    assert np.allclose(sim1.data[dp], sim2.data[dp], **tols)
    assert not np.allclose(sim1.data[ep], sim2.data[ep], **tols)


def test_present_input(Simulator, rng):
    n = 5
    c, ni, nj = 3, 8, 8
    images = rng.normal(size=(n, c, ni, nj))
    pres_time = 0.1

    model = nengo.Network()
    with model:
        u = nengo.Node(nengo.processes.PresentInput(images, pres_time))
        up = nengo.Probe(u)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    i = (np.floor((t - sim.dt) / pres_time + 1e-7) % n).astype(int)
    y = sim.data[up].reshape(len(t), c, ni, nj)
    for k, [ii, image] in enumerate(zip(i, y)):
        assert np.allclose(image, images[ii], rtol=1e-4, atol=1e-7), (k, ii)
