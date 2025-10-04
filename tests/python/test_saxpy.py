import reprospect

class TestSaxpySass(reprospect.Case):

    def test_ninst(self):
        cuobjdump = self.cuobjdump

        decoder = reprospect.Decoder(code = cuobjdump.sass)

        expt_ninst = 32

        self.assertEqual(len(decoder.instructions), expt_ninst)

if __name__ == '__main__':
    reprospect.main()
