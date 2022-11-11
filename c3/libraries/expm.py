import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from math import factorial


@tf.function
def matrix_exponential(matrix:tf.Tensor):
    """Computes the matrix exponential of one or more square matrices.
    $$exp(A) = \sum_{n=0}^\infty A^n/n!$$
    The exponential is computed using a combination of the scaling and squaring
    method and the Pade approximation. Details can be found in:
    Nicholas J. Higham, "A New Scaling and Squaring Algorithm for the
    Matrix Exponential," SIAM J. Matrix Anal. Applic., 31(3):970-989, 2009.
    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. The output is a tensor of the same shape as the input
    containing the exponential for all input submatrices `[..., :, :]`.
    This method implements a customized gradient computation based on the scaling and
    squaring method, with more details found in: Nicholas J. Higham, "Computing the 
    Fréchet Derivative of the Matrix Exponential, with an application to Condition
    Number Estimation," SIAM J. Matrix Anal. Applic., 30(4):1639-1657, 2008.
    Args:
    input: A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`, or
        `complex128` with shape `[..., M, M]`.
    dt: A 'int'.
    Returns:
    the matrix exponential of the input.
    Raises:
    ValueError: An unsupported type is provided as input.
    """
    if matrix.shape[-2:] == [0, 0]:
        return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
        batch_shape = array_ops.shape(matrix)[:-2]
    
    # reshaping the batch makes the where statements work better
    matrix = array_ops.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    l1_norm = math_ops.reduce_max(
        math_ops.reduce_sum(
            math_ops.abs(matrix),
            axis=array_ops.size(array_ops.shape(matrix)) - 2),
        axis=-1)[..., array_ops.newaxis, array_ops.newaxis]
            
    coeff = [1/100800, 1/10059033600, 1/4487938430976000, 1/5914384781877411840000, 1/113250775606021113483283660800000000]
    coeff = [constant_op.constant(x, matrix.dtype) for x in coeff]
           
    l1_norm = tf.cast(l1_norm, dtype=tf.dtypes.float32)
    maxnorm = tf.constant(4.5)
    squarings = math_ops.maximum(
        math_ops.floor(
            math_ops.log(l1_norm / maxnorm) / math_ops.log(tf.constant(2.0))), 0)
    
    matrix_2 = math_ops.matmul(matrix, matrix)
    matrix_4 = math_ops.matmul(matrix_2, matrix_2)
    matrix_6 = math_ops.matmul(matrix_2, matrix_4)
    
    b = [64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
        33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0]
    b = [constant_op.constant(x, matrix.dtype) for x in b]

    ident = linalg_ops.eye(
        array_ops.shape(matrix)[-2],
        batch_shape=array_ops.shape(matrix)[:-2],
        dtype=matrix.dtype)
    W1 = matrix_6 + b[11] * matrix_4 + b[9] * matrix_2
    W = (
        math_ops.matmul(matrix_6, W1) +
        b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident)
    u13 = math_ops.matmul(matrix, W)
    Z1 = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
    v13 = (
        math_ops.matmul(matrix_6, Z1) + b[6] * matrix_6 + b[4] * matrix_4 +
        b[2] * matrix_2 + b[0] * ident)

    is_finite = math_ops.is_finite(math_ops.reduce_max(l1_norm))
    nan = constant_op.constant(np.nan, matrix.dtype)
    um = -u13 + v13
    us = u13 + v13
    matrix_s = control_flow_ops.cond(
        is_finite, lambda: linalg_ops.matrix_solve(um, us),
        lambda: array_ops.fill(array_ops.shape(matrix), nan))
    
    max_squarings = math_ops.reduce_max(squarings)
    i = tf.constant(0.0)

    def cond(i, _):
      return control_flow_ops.cond(is_finite,
                                   lambda: math_ops.less(i, max_squarings),
                                   lambda: constant_op.constant(False))

    def body(i, r):
      return i + 1, array_ops.where_v2(
          math_ops.less(i, squarings), math_ops.matmul(r, r), r)

    _, result = control_flow_ops.while_loop(cond, body, [i, matrix_s])
    result = array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))

    def grad(upstream):
        E = upstream
        M2 = math_ops.matmul(matrix, E) + math_ops.matmul(E, matrix)
        M4 = math_ops.matmul(matrix_2, M2) + math_ops.matmul(M2, matrix_2)
        M6 = math_ops.matmul(matrix_4, M2) + math_ops.matmul(M4, matrix_2)
        Lw = (
            math_ops.matmul(matrix_6, M6 + b[11] * M4 + b[9] * M2)
            + math_ops.matmul(M6, W1) + b[7] * M6 + b[5] * M4 + b[3] * M2
        )
        Lu = math_ops.matmul(matrix, Lw) + math_ops.matmul(E, W)
        Lv = (
            math_ops.matmul(matrix_6, b[12] * M6 + b[10] * M4 + b[8] * M2)
            + math_ops.matmul(M6, Z1) + b[6] * M6 + b[4] * M4 + b[2] * M2
        )
        L = Lu + Lv + math_ops.matmul(Lu - Lv, matrix_s)
        L = control_flow_ops.cond(
            is_finite, lambda: linalg_ops.matrix_solve(um, L),
            lambda: array_ops.fill(array_ops.shape(matrix), nan))

        i = tf.constant(0.0)
        
        def cond(i, *_):
          return control_flow_ops.cond(is_finite,
                                   lambda: math_ops.less(i, max_squarings),
                                   lambda: constant_op.constant(False))

        def body(i, l, r):
            return i + 1, array_ops.where_v2(
                math_ops.less(i, squarings), math_ops.matmul(r, l) + math_ops.matmul(l, r), l), r

        _, result, mat = control_flow_ops.while_loop(cond, body, [i, L, matrix_s])
        return tf.math.multiply(result, upstream)
    return result

@tf.function
def matrix_exponential_taylor_18(matrix):
    if matrix.shape[-2:] == [0, 0]:
        return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
        batch_shape = array_ops.shape(matrix)[:-2]
    a = [0, -0.10036558103014462001, -0.00802924648241156960, -0.00089213849804572995]
    b1 = [0, 0.39784974949964507614, 1.36783778460411719922, 0.49828962252538267755, -0.00063789819459472330]
    b2 = [-10.9676396052962062593, 1.68015813878906197182, 0.05717798464788655127, -0.00698210122488052084, 0.00003349750170860705]
    b3 = [-0.09043168323908105619, -0.06764045190713819075, 0.06759613017704596460, 0.02955525704293155274, -0.00001391802575160607]
    b4 = [0, 0, -0.09233646193671185927, -0.01693649390020817171, -0.00001400867981820361]
    
    a = [constant_op.constant(x, matrix.dtype) for x in a]
    b1 = [constant_op.constant(x, matrix.dtype) for x in b1]
    b2 = [constant_op.constant(x, matrix.dtype) for x in b2]
    b3 = [constant_op.constant(x, matrix.dtype) for x in b3]
    b4 = [constant_op.constant(x, matrix.dtype) for x in b4]

    ident = linalg_ops.eye(
        array_ops.shape(matrix)[-2],
        batch_shape=array_ops.shape(matrix)[:-2],
        dtype=matrix.dtype)

    # reshaping the batch makes the where statements work better
    matrix = array_ops.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    matrix_2 = math_ops.matmul(matrix, matrix)
    matrix_3 = math_ops.matmul(matrix_2, matrix)
    matrix_6 = math_ops.matmul(matrix_3, matrix_3)
    matrix_9 = (
        math_ops.matmul(a[1] * matrix + a[2] * matrix_2 + a[3] * matrix_3, b4[0] * ident + b4[1] * matrix + b4[2] * matrix_2 + b4[3] * matrix_3 + b4[4] * matrix_6)
        + b3[0] * ident + b3[1] * matrix + b3[2] * matrix_2 + b3[3] * matrix_3 + b3[4] * matrix_6
    )
    mat = math_ops.matmul(b2[0] * ident + b2[1] * matrix + b2[2] * matrix_2 + b2[3] * matrix_3 + b2[4] * matrix_6 + matrix_9, matrix_9) + b1[1] * matrix + b1[2] * matrix_2 + b1[3] * matrix_3 + b1[4] * matrix_6
    return mat

@tf.function
def matrix_exponential_taylor_12(matrix):
    if matrix.shape[-2:] == [0, 0]:
        return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
        batch_shape = array_ops.shape(matrix)[:-2]
    a1 = [-0.01860232051462055322, 4.60000000000000000000, 0.21169311829980944294, 0]
    a2 = [-0.00500702322573317730, 0.99287510353848683614, 0.15822438471572672537, -0.13181061013830184015]
    a3 = [-0.57342012296052226390, -0.13244556105279963884, 0.16563516943672741501, -0.02027855540589259079]
    a4 = [-0.13339969394389205970, 0.00172990000000000000, 0.01078627793157924250, -0.00675951846863086359]
    
    a1 = [constant_op.constant(x, matrix.dtype) for x in a1]
    a2 = [constant_op.constant(x, matrix.dtype) for x in a2]
    a3 = [constant_op.constant(x, matrix.dtype) for x in a3]
    a4 = [constant_op.constant(x, matrix.dtype) for x in a4]

    ident = linalg_ops.eye(
        array_ops.shape(matrix)[-2],
        batch_shape=array_ops.shape(matrix)[:-2],
        dtype=matrix.dtype)

    # reshaping the batch makes the where statements work better
    matrix = array_ops.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    matrix_2 = math_ops.matmul(matrix, matrix)
    matrix_3 = math_ops.matmul(matrix_2, matrix)
    b4 = a1[3] * ident + a2[3] * matrix + a3[3] * matrix_2 + a4[3] * matrix_3
    matrix_6 = (
        math_ops.matmul(b4, b4) + a1[2] * ident + a2[2] * matrix + a3[2] * matrix_2 + a4[2] * matrix_3
    )
    mat = math_ops.matmul(a1[1] * ident + a2[1] * matrix + a3[1] * matrix_2 + a4[1] * matrix_3 + matrix_6, matrix_6) + a1[0] * ident + a2[0] * matrix + a3[0] * matrix_2 + a4[0] * matrix_3
    return mat
