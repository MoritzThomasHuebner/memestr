def joint_pdf(m_1, m_2, alpha, m_min, m_max, M_max, const=None):
    r"""
    Computes the probability density for the joint mass distribution
    :math:`p(m_1, m_2)` defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       \frac{m_1^{-\alpha}}{m_1 - m_{\mathrm{min}}}

    Computes the normalization constant using :func:`pdf_const` if not provided
    by the ``const`` keyword argument.
    """

    import numpy

    # Ensure everything is a numpy array.
    m_1 = numpy.asarray(m_1)
    m_2 = numpy.asarray(m_2)
    m_min = numpy.asarray(m_min)
    m_max = numpy.asarray(m_max)
    M_max = numpy.asarray(M_max)

    # Array of booleans determining which values of ``m_1`` and ``m_2`` do not
    # correspond to zero probability.
    cond = (
        (m_1 >= m_2) &
        (m_1 > m_min) & (m_2 >= m_min) &
        (m_1 <= m_max) &
        (m_1 + m_2 <= M_max)
    )

    # Compute the normalization constant if it has not been pre-computed.
    if const is None:
        const = pdf_const(alpha, m_min, m_max, M_max)

    # Compute the value of the PDF everywhere, though this will be invalid in
    # regions where the PDF is zero due to conditions. Those values will not be
    # used and need not be valid.
    pdf_support = const * numpy.power(m_1, -alpha) / (m_1 - m_min)

    # Return the value PDF wherever it is non-zero according to the condition,
    # and return zero everywhere else.
    return numpy.where(cond, pdf_support, 0.0)


def marginal_pdf(m1, alpha, m_min, m_max, M_max, const=None):
    r"""
    Computes the probability density for the marginal mass distribution
    :math:`p(m_1)` defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       m_1^{-\alpha} \,
       \frac{\min(m_1, M_{\mathrm{max}}-m_1) - m_{\mathrm{min}}}
            {m_1 - m_{\mathrm{min}}}

    Computes the normalization constant using :func:`pdf_const` if not provided
    by the ``const`` keyword argument.
    """

    import numpy
    from six.moves import zip

    m1 = numpy.asarray(m1)
    alphas, m_mins, m_maxs = upcast_scalars((alpha, m_min, m_max))

    pdf = numpy.zeros((len(alphas), len(m1)), dtype=m1.dtype)

    for i, (alpha, m_min, m_max) in enumerate(zip(alphas, m_mins, m_maxs)):
        support = (m1 > m_min) & (m1 <= m_max)
        m1_support = m1[support]

        if const is None:
            const = pdf_const(alpha, m_min, m_max, M_max)

        pl_term = numpy.power(m1_support, -alpha)
        cutoff_term = (
            (numpy.minimum(m1_support, M_max-m1_support) - m_min) /
            (m1_support - m_min)
        )

        pdf[i, support] = const * pl_term * cutoff_term

    return pdf

def pdf_const(alpha, m_min, m_max, M_max):
    r"""
    Computes the normalization constant
    :math:`C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}})`, according
    to the derivation given in [T1700479]_.

    .. [T1700479]
        Normalization constant in power law BBH mass distribution model,
        Daniel Wysocki and Richard O'Shaughnessy,
        `LIGO-T1700479 <https://dcc.ligo.org/LIGO-T1700479>`_
    """

    import numpy

    beta = 1 - alpha

    # Special case for beta = 0
    if beta == 0:
        return _pdf_const_special(m_min, m_max, M_max)
    else:
        return _pdf_const_nonspecial(beta, m_min, m_max, M_max)


def _pdf_const_special(m_min, m_max, M_max):
    import numpy

    if m_max > 0.5*M_max:
        A = numpy.log(0.5) + numpy.log(M_max) - numpy.log(m_min)

        B1 = (
            (M_max - 2*m_min) *
            numpy.log((m_max - m_min) / (0.5*M_max - m_min))
        )
        B2 = (M_max - m_min) * numpy.log(0.5 * M_max / m_max)
        B = (B1 + B2) / m_min
    else:
        A = numpy.log(m_max) - numpy.log(m_min)
        B = 0

    return numpy.reciprocal(A + B)


def _pdf_const_nonspecial(beta, m_min, m_max, M_max):
    import numpy
    from mpmath import hyp2f1

    eps = 1e-7
    if numpy.int32(beta) == beta:
        const_plus = _pdf_const_nonspecial(beta+eps, m_min, m_max, M_max)
        const_minus = _pdf_const_nonspecial(beta-eps, m_min, m_max, M_max)
        return 0.5 * (const_plus + const_minus)

    if m_max > 0.5*M_max:
        A = (numpy.power(0.5*M_max, beta) - numpy.power(m_min, beta)) / beta

        B1a = (
            numpy.power(0.5*M_max, beta) *
            hyp2f1(1, beta, 1+beta, 0.5*M_max/m_min)
        )
        B1b = (
            numpy.power(m_max, beta) *
            hyp2f1(1, beta, 1+beta, m_max/m_min)
        )
        B1 = (M_max - 2*m_min) * (B1a - B1b) / m_min

        B2 = numpy.power(0.5*M_max, beta) - numpy.power(m_max, beta)

        B = numpy.float64((B1 + B2).real) / beta
    else:
        A = (numpy.power(m_max, beta) - numpy.power(m_min, beta)) / beta
        B = 0

    return numpy.reciprocal(A + B)

def upcast_scalars(arrays):
    import numpy
    dims = [numpy.ndim(arr) for arr in arrays]
    shape = numpy.shape(arrays[numpy.argmax(dims)])

    if shape == ():
        shape = (1,)

    result = []
    for arr in arrays:
        if numpy.shape(arr) == shape:
            result.append(numpy.asarray(arr))
        else:
            result.append(numpy.tile(arr, shape))

    return result
