"""Small offline checks for formats and resumable download handling."""
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from data import download, read_bin, read_vectors, write_bin

class Response(io.BytesIO):
    def __init__(self,body,status,headers):
        super().__init__(body);self.status=status;self.headers=headers

class DataTests(unittest.TestCase):
    def test_formats_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=np.arange(15,dtype='<f4').reshape(5,3)
            write_bin(root/'x.fbin',a);np.testing.assert_array_equal(read_bin(root/'x.fbin'),a)
            with (root/'x.fvecs').open('wb') as f:
                for row in a:f.write(struct.pack('<I',3));row.tofile(f)
            np.testing.assert_array_equal(read_vectors(root/'x.fvecs'),a)
            (root/'bad.fbin').write_bytes((root/'x.fbin').read_bytes()[:-1])
            with self.assertRaises(ValueError):read_bin(root/'bad.fbin')

    def test_http_resume_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'x';part=Path(d)/'x.part';part.write_bytes(b'abcd')
            with patch('urllib.request.urlopen',return_value=Response(b'efgh',206,{'Content-Range':'bytes 4-7/8','Content-Length':'4'})) as request:
                download('https://example.invalid/x',out,8)
                self.assertEqual(request.call_args.args[0].get_header('Range'),'bytes=4-7')
            self.assertEqual(out.read_bytes(),b'abcdefgh')
            out.unlink();part.write_bytes(b'abcd')
            with patch('urllib.request.urlopen',return_value=Response(b'abcdefgh',200,{'Content-Length':'8'})):
                with self.assertRaises(RuntimeError):download('https://example.invalid/x',out,8)
            self.assertEqual(part.read_bytes(),b'abcd')

    def test_ground_truth_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=np.array([[0,2],[1,3]],dtype='<u4')
            with (root/'download').open('wb') as f:
                f.write(struct.pack('<II',2,2));ids.tofile(f);np.zeros((2,2),dtype='<f4').tofile(f)
            subprocess.run([sys.executable,str(Path(__file__).with_name('data.py')),'convert','--format','bigann-gt','--input',str(root/'download'),'--output',str(root/'gt.ibin')],check=True)
            np.testing.assert_array_equal(read_bin(root/'gt.ibin','uint32'),ids)

if __name__=='__main__':unittest.main()
