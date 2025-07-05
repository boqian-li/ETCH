import math
from vgtk.point3d import PointSet
from vgtk.pc import PointCloud
import vgtk.transform.operator as op


# define meta transform function decorator
def pointmethod(f):
    def func(x):
        if isinstance(x, PointSet):
            xyz = x.data
            return PointSet(f(xyz))
        elif isinstance(x, PointCloud):
            xyz, feats = x.data
            return PointCloud(f(xyz), feats)
        else:
            raise ValueError()

    return func


class Transform():
    def __init__(self, T):
        '''
        T: [(b, ){3,4}, {3,4}]
        '''
        self._T = T

    @property
    def data(self):
        return self._T

    @classmethod
    def from_Rt(cls, R, t):
        '''
        R: [(b, )3, 3]
        t: [(b, )3]
        '''
        if R.ndim == 3 and t.ndim == 2:
            assert R.shape[0] == t.shape[0], 'batch mismatch'
        elif R.ndim == 3:
            t = t[None].repeat(R.shape[0], 1)
        elif t.ndim == 2:
            R = R[None].repeat(t.shape[0], 1, 1)
        T = torch.cat((R, t.unsqueeze(-2)), dim=-2).contiguous()
        return cls(T)

    @classmethod
    def from_eular_angle(cls, angles):
        '''
        angles: [(b, )3]
        '''
        T = op.R_from_euler_angle(angles)
        return cls(T)

    @classmethod
    def from_quaternion(cls, q):
        '''
        q: [(b, )4]
        '''
        raise NotImplementedError()

    @classmethod
    def from_exp_matrix(cls, w):
        '''
        w: [(b, )3]
        '''
        raise NotImplementedError()

    @pointmethod
    def transform(self, x):
        '''
            x: [(b, ){3/4}, n]
        Return:
            [(b, ){3/4}, n]
        '''
        if self._T.shape[-1] == 3:
            x = x.from_hom()
            return op.rotate(x.data, self._T)
        else:
            x = x.to_hom()
            return op.transform(x.data, self._T)

    def __call__(self, x):
        return self.transform(x)
