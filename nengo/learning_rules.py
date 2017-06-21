import warnings

from nengo.exceptions import ValidationError
from nengo.params import IntParam, FrozenObject, NumberParam, Parameter
from nengo.utils.compat import is_iterable, is_string, itervalues


class LearningRuleTypeSizeInParam(IntParam):
    valid_strings = ('pre', 'post', 'mid', 'pre_state', 'post_state')

    def coerce(self, instance, size_in):
        if is_string(size_in):
            if size_in not in self.valid_strings:
                raise ValidationError(
                    "%r is not a valid string value (must be one of %s)"
                    % (size_in, self.strings), attr=self.name, obj=instance)
            return size_in
        else:
            return super(LearningRuleTypeSizeInParam, self).coerce(
                instance, size_in)  # IntParam validation


class LearningRuleType(FrozenObject):
    """Base class for all learning rule objects.

    To use a learning rule, pass it as a ``learning_rule_type`` keyword
    argument to the `~nengo.Connection` on which you want to do learning.

    Each learning rule exposes two important pieces of metadata that the
    builder uses to determine what information should be stored.

    The ``size_in`` is the dimensionality of the incoming error signal. It
    can either take an integer or one of the following string values:

    * ``'pre'``: vector error signal in pre-object space
    * ``'post'``: vector error signal in post-object space
    * ``'mid'``: vector error signal in the ``conn.size_mid`` space
    * ``'pre_state'``: vector error signal in pre-synaptic ensemble space
    * ``'post_state'``: vector error signal in pre-synaptic ensemble space

    The difference between ``'post_state'`` and ``'post'`` is that with the
    former, if a ``Neurons`` object is passed, it will use the dimensionality
    of the corresponding ``Ensemble``, whereas the latter simply uses the
    ``post`` object ``size_in``. Similarly with ``'pre_state'`` and ``'pre'``.

    The ``modifies`` attribute denotes the signal targeted by the rule.
    Options are:

    * ``'encoders'``
    * ``'decoders'``
    * ``'weights'``

    Parameters
    ----------
    learning_rate : float, optional (Default: 1e-6)
        A scalar indicating the rate at which ``modifies`` will be adjusted.
    size_in : int, str, optional (Default: 0)
        Dimensionality of the error signal (see above).

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which ``modifies`` will be adjusted.
    size_in : int, str
        Dimensionality of the error signal.
    modifies : str
        The signal targeted by the learning rule.
    """

    modifies = None
    probeable = ()

    learning_rate = NumberParam('learning_rate', low=0, low_open=True)
    size_in = LearningRuleTypeSizeInParam('size_in', low=0)

    def __init__(self, learning_rate=1e-6, size_in=0, clip_type=None, decay_rate_x_dt=0.0):
        '''
            clip_type can be one of None, 'clip<0', 'clip>0'
            decay_rate_x_dt is a float that multiplies weight at every time step after weight updation
        '''
        super(LearningRuleType, self).__init__()
        self.learning_rate = learning_rate
        self.size_in = size_in
        ## convert clip_type internally to an integer for faster comparison
        if clip_type is None:
            self.clip_type = 0
        elif clip_type == 'clip<0':
            self.clip_type = 1
        elif clip_type == 'clip>0':
            self.clip_type = 2
        else:
            warnings.warn("This %s clip_type is not supported."
                        "Reverting to no clipping." % str(clip_type))
        self.decay_rate_x_dt = decay_rate_x_dt

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ", ".join(self._argreprs))

    @property
    def _argreprs(self):
        return (["learning_rate=%g" % self.learning_rate]
                if self.learning_rate != 1e-6 else [])


class PES(LearningRuleType):
    """Prescribed Error Sensitivity learning rule.

    Modifies a connection's decoders to minimize an error signal provided
    through a connection to the connection's learning rule.

    Parameters
    ----------
    learning_rate : float, optional (Default: 1e-4)
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    integral_tau : float or None
        tau for integrating the delta w; if None, no integration.
    decay_rate_x_dt : float
        decay rate*dt for the weights
    """

    modifies = 'decoders'
    probeable = ('error', 'correction', 'activities', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)

    def __init__(self, learning_rate=1e-4, pre_tau=0.005,
                    clip_type=None, decay_rate_x_dt=0.0, integral_tau=None):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        self.integral_tau = integral_tau
        super(PES, self).__init__(learning_rate, size_in='post_state', clip_type, decay_rate_x_dt)

    @property
    def _argreprs(self):
        args = []
        if self.learning_rate != 1e-4:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.pre_tau != 0.005:
            args.append("pre_tau=%g" % self.pre_tau)
        return args

class VoltageRule(LearningRuleType):
    """ dw/dt = learning_rate * post_voltage * pre_spiking_low_pass

    Parameters
    ----------
    pre_tau : float, optional
        Filter constant on activities of neurons in pre population.
        Defaults to 0.005.
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.

    Attributes
    ----------
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    learning_rate : float
        The given learning rate.
    integral_tau : float or None
        tau for integrating the delta w; if None, no integration.
    decay_rate_x_dt : float
        decay rate*dt for the weights
    """

    pre_tau = NumberParam(low=0, low_open=True)

    error_type = 'decoded'
    modifies = 'decoders'
    probeable = ['correction', 'activities', 'delta']

    def __init__(self, learning_rate=1e-4, pre_tau=0.005,
                    clipType=None, decay_rate_x_dt=0.0, integral_tau=None):
        if learning_rate >= 1.0:
            warnings.warn("This learning rate is very high, and can result "
                          "in floating point errors from too much current.")
        self.pre_tau = pre_tau
        self.integral_tau = integral_tau
        super(VoltageRule, self).__init__(learning_rate, clipType, decay_rate_x_dt)

    @property
    def _argreprs(self):
        args = []
        if self.learning_rate != 1e-4:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        return args

class BCM(LearningRuleType):
    """Bienenstock-Cooper-Munroe learning rule.

    Modifies connection weights as a function of the presynaptic activity
    and the difference between the postsynaptic activity and the average
    postsynaptic activity.

    Notes
    -----
    The BCM rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the BCM rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the BCM rule by ``1 / post.n_neurons``.

    Parameters
    ----------
    theta_tau : float, optional (Default: 1.0)
        A scalar indicating the time constant for theta integration.
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional (Default: None)
        Filter constant on activities of neurons in post population.
        If None, post_tau will be the same as pre_tau.
    learning_rate : float, optional (Default: 1e-9)
        A scalar indicating the rate at which weights will be adjusted.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    theta_tau : float
        A scalar indicating the time constant for theta integration.
    """

    modifies = 'weights'
    probeable = ('theta', 'pre_filtered', 'post_filtered', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    theta_tau = NumberParam('theta_tau', low=0, low_open=True)

    def __init__(self, pre_tau=0.005, post_tau=None, theta_tau=1.0,
                 theta=None, learning_rate=1e-9):
        self.theta = theta
        self.theta_tau = theta_tau
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        super(BCM, self).__init__(learning_rate, size_in=0)

    @property
    def _argreprs(self):
        args = []
        if self.pre_tau != 0.005:
            args.append("pre_tau=%g" % self.pre_tau)
        if self.post_tau != self.pre_tau:
            args.append("post_tau=%g" % self.post_tau)
        if self.theta_tau != 1.0:
            args.append("theta_tau=%g" % self.theta_tau)
        if self.learning_rate != 1e-9:
            args.append("learning_rate=%g" % self.learning_rate)
        return args

class InhVSG(LearningRuleType):
    """Vogels-Sprekeler-Gerstner learning rule

    Modifies connection weights.

    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which decoders will be adjusted.
        Defaults to 1e-5.
    theta : float, optional
        A scalar indicating the desired firing rate.
    pre_tau : float, optional
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional
        Filter constant on activities of neurons in post population.

    Attributes
    ----------
    learning_rate : float
        The given learning rate.
    theta : float
        A scalar indicating the desired firing rate.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    pre_tau = NumberParam(low=0, low_open=True)
    post_tau = NumberParam(low=0, low_open=True)
    theta = NumberParam(low=0, low_open=True)

    error_type = 'scalar' # needed to return learning_rule.size_in=1 (see LearningRule in connection.py)
    modifies = 'weights'
    probeable = ['pre_filtered', 'post_filtered', 'delta']

    def __init__(self, pre_tau=0.005, post_tau=None, theta=100.0,
                 learning_rate=1e-9, clipType=None, decay_rate_x_dt=0.0):
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        self.theta = theta
        super(InhVSG, self).__init__(learning_rate, clipType, decay_rate_x_dt)

    @property
    def _argreprs(self):
        args = []
        if self.pre_tau != 0.005:
            args.append("pre_tau=%f" % self.pre_tau)
        if self.post_tau != self.pre_tau:
            args.append("post_tau=%f" % self.post_tau)
        if self.theta != 3.0:
            args.append("theta=%f" % self.theta)
        if self.learning_rate != 1e-9:
            args.append("learning_rate=%g" % self.learning_rate)
        return args

class Oja(LearningRuleType):
    """Oja learning rule.

    Modifies connection weights according to the Hebbian Oja rule, which
    augments typicaly Hebbian coactivity with a "forgetting" term that is
    proportional to the weight of the connection and the square of the
    postsynaptic activity.

    Notes
    -----
    The Oja rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the Oja rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the Oja rule by ``1 / post.n_neurons``.

    Parameters
    ----------
    pre_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in pre population.
    post_tau : float, optional (Default: None)
        Filter constant on activities of neurons in post population.
        If None, post_tau will be the same as pre_tau.
    beta : float, optional (Default: 1.0)
        A scalar weight on the forgetting term.
    learning_rate : float, optional (Default: 1e-6)
        A scalar indicating the rate at which weights will be adjusted.

    Attributes
    ----------
    beta : float
        A scalar weight on the forgetting term.
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    pre_tau : float
        Filter constant on activities of neurons in pre population.
    """

    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    pre_tau = NumberParam('pre_tau', low=0, low_open=True)
    post_tau = NumberParam('post_tau', low=0, low_open=True)
    beta = NumberParam('beta', low=0)

    def __init__(self, pre_tau=0.005, post_tau=None, beta=1.0,
                 learning_rate=1e-6):
        self.pre_tau = pre_tau
        self.post_tau = post_tau if post_tau is not None else pre_tau
        self.beta = beta
        super(Oja, self).__init__(learning_rate, size_in=0)

    @property
    def _argreprs(self):
        args = []
        if self.pre_tau != 0.005:
            args.append("pre_tau=%g" % self.pre_tau)
        if self.post_tau != self.pre_tau:
            args.append("post_tau=%g" % self.post_tau)
        if self.beta != 1.0:
            args.append("beta=%g" % self.beta)
        if self.learning_rate != 1e-6:
            args.append("learning_rate=%g" % self.learning_rate)
        return args


class Voja(LearningRuleType):
    """Vector Oja learning rule.

    Modifies an ensemble's encoders to be selective to its inputs.

    A connection to the learning rule will provide a scalar weight for the
    learning rate, minus 1. For instance, 0 is normal learning, -1 is no
    learning, and less than -1 causes anti-learning or "forgetting".

    Parameters
    ----------
    post_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in post population.
    learning_rate : float, optional (Default: 1e-2)
        A scalar indicating the rate at which encoders will be adjusted.

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which encoders will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    modifies = 'encoders'
    probeable = ('post_filtered', 'scaled_encoders', 'delta')

    post_tau = NumberParam('post_tau', low=0, low_open=True, optional=True)

    def __init__(self, post_tau=0.005, learning_rate=1e-2):
        self.post_tau = post_tau
        super(Voja, self).__init__(learning_rate, size_in=1)

    @property
    def _argreprs(self):
        args = []
        if self.post_tau is None:
            args.append("post_tau=%s" % self.post_tau)
        elif self.post_tau != 0.005:
            args.append("post_tau=%g" % self.post_tau)
        if self.learning_rate != 1e-2:
            args.append("learning_rate=%g" % self.learning_rate)
        return args


class LearningRuleTypeParam(Parameter):
    def check_rule(self, instance, rule):
        if not isinstance(rule, LearningRuleType):
            raise ValidationError(
                "'%s' must be a learning rule type or a dict or "
                "list of such types." % rule, attr=self.name, obj=instance)
        if rule.modifies not in ('encoders', 'decoders', 'weights'):
            raise ValidationError("Unrecognized target %r" % rule.modifies,
                                  attr=self.name, obj=instance)

    def coerce(self, instance, rule):
        if is_iterable(rule):
            for r in (itervalues(rule) if isinstance(rule, dict) else rule):
                self.check_rule(instance, r)
        elif rule is not None:
            self.check_rule(instance, rule)
        return super(LearningRuleTypeParam, self).coerce(instance, rule)
