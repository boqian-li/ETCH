import torch
# import vgtk.point3d as p3tk

def pointmethod(f):
    def func(x):
        return PointSet(f(x._p))
    return func

def samplemethod(f):
    def func(x):
        idx, p = f(x._p)
        return idx, PointSet(p)
    return func

class PointSet():
    def __init__(self, p):
        '''
        p: [(b, ){3/4}, n]
        '''
        self._p = p
        # if len(self._p.shape) == 2:
        #     self._p = self._p[None]

    @property
    def is_hom(self):
        return self._p.shape[-2] == 4

    @property
    def n_batch(self):
        return self._p.shape[0]

    @property
    def n_point(self):
        return self._p.shape[-1]

    @property
    def device(self):
        return self._p.device

    @property
    def data(self):
        return self._p

    ## homography

    @pointmethod
    def to_hom(self):
        if self.is_hom:
            return self._p
        ones = torch.ones(self.n_batch, 1, self.n_point).to(self.device)
        return torch.cat((self._p, ones), dim=-2)

    @pointmethod
    def from_hom(self):
        if not self.is_hom:
            return self._p
        return self._p[..., :3, :]

    ## sampling

    # @samplemethod
    # def furthest_sample(self, n_sample, lazy_sample=False):
    #     return p3tk.furthest_sample(self._p, n_sample, lazy_sample)

    # @samplemethod
    # def ball_query(self, q, r, n_sample):
    #     '''
    #     Args
    #         q: [b, 3, m]
    #     '''
    #     return p3tk.ball_query(self._p, q, r, n_sample)

    # ## normalize

    # @pointmethod
    # def centralize(self):
    #     return p3tk.centralize(self._p)

    # @pointmethod
    # def normalize(self):
    #     return p3tk.normalize(self._p)
    # '''
