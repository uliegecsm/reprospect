import reprospect

from reprospect.tools.ncu import Metric

class TestSaxpySass(reprospect.Case):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.cuobjdump, cls.decoder = cls.cuobjdump_and_decode()

    def test_ninst(self):
        expt_ninst = 32

        self.assertEqual(len(self.decoder.instructions), expt_ninst)

class TestSaxpyNcu(reprospect.Case):

    METRICS = [
        Metric(name = 'launch__registers_per_thread_allocated')
    ]

    NVTX_CAPTURE = 'application-domain@outer-useless-range'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.report = cls.ncu(metrics = cls.METRICS, nvtx_capture = cls.NVTX_CAPTURE)
    
    def test_nranges(self):
        self.assertEqual(self.report.num_ranges(), 1)

if __name__ == '__main__':
    reprospect.main()
