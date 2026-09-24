"""Standard-library release tests: no weights, CUDA or private data required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_runner', ROOT / 'run_release.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.mapping = self.base / 'mapping.json'
        self.case = dict(source_prompt='a red car', target_prompt='a blue car',
                         points=[[12, 15], [30, 40]], mask_path='hint.png',
                         image_path='input.png', drag_type='2D-Rigid')

    def write(self, data):
        self.mapping.write_text(json.dumps(data))

    def args(self, mode='joint', extra=()):
        return runner.parse_args(['--mode', mode, '--data-root', str(self.base),
            '--mapping', str(self.mapping), '--output', str(self.base / 'output'), *extra])

    def test_primary_explicit_settings(self):
        for mode, steps, strength, guidance in [('text',12,1.0,2.1),('drag',17,.75,1.),('joint',15,.7,1.5)]:
            cmd, values = runner.build_command(self.args(mode), self.mapping)
            self.assertEqual(values['steps'], steps)
            self.assertEqual(values['strength'], strength)
            self.assertEqual(values['guidance_t'], guidance)
            self.assertIn('--no-use_saved_influence_range', cmd)
            self.assertIn('--no-ref_kv_injection' if mode=='text' else '--ref_kv_injection', cmd)

    def test_seed_override(self):
        _, values = runner.build_command(self.args(extra=['--seed','123']), self.mapping)
        self.assertEqual(values['seed'],123)

    def test_nested_author_view(self):
        official = {**self.case, 'mask_path':'official.png'}
        self.write({'case':{'source':official,'modified':self.case}})
        cases = runner.load_cases(self.mapping,'joint')
        self.assertEqual(runner.selected_entry(cases['case'],'joint')['mask_path'],'hint.png')
        self.assertEqual(runner.selected_entry(cases['case'],'text')['mask_path'],'official.png')

    def test_missing_hint_fails(self):
        del self.case['mask_path']
        self.write({'case':self.case})
        with self.assertRaisesRegex(ValueError,'author mask'):
            runner.load_cases(self.mapping,'joint')

    def test_missing_target_fails(self):
        del self.case['target_prompt']
        self.write({'case':self.case})
        with self.assertRaisesRegex(ValueError,'target prompt'):
            runner.load_cases(self.mapping,'joint')

    def test_invalid_points(self):
        for points in ([], [[1,2]], [[1,2],[3,float('nan')]], [[1,2,3],[3,4]], [[True,2],[3,4]]):
            self.case['points']=points
            self.write({'case':self.case})
            with self.assertRaises(ValueError):
                runner.load_cases(self.mapping,'joint')

    def test_unsafe_case_ids(self):
        for key in ('../escape','..','/absolute','a/b','a\\b'):
            self.write({key:self.case})
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                runner.load_cases(self.mapping,'joint')

    def test_empty_mapping(self):
        self.write({})
        with self.assertRaises(ValueError):
            runner.load_cases(self.mapping,'joint')

    def test_missing_asset_fails(self):
        self.write({'case':self.case})
        with self.assertRaises(FileNotFoundError):
            runner.load_cases(self.mapping,'joint',check_assets=True,data_root=self.base)

    def test_dry_run_does_not_write(self):
        self.write({'case':self.case})
        args=['--mode','joint','--data-root',str(self.base),'--mapping',str(self.mapping),
              '--output',str(self.base/'output'),'--dry-run']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(args),0)
        self.assertFalse((self.base/'output').exists())

    def test_limit_is_explicit(self):
        self.write({'one':self.case,'two':self.case})
        self.assertEqual(list(runner.load_cases(self.mapping,'joint',1)),['one'])

    def test_nonempty_output_preserved(self):
        self.write({'case':self.case})
        for name in ('input.png','hint.png'):
            (self.base/name).write_bytes(b'fixture')
        (self.base/'output').mkdir()
        protected=self.base/'output'/'existing.txt'
        protected.write_text('preserve')
        with self.assertRaisesRegex(ValueError,'not empty'):
            runner.main(['--mode','joint','--data-root',str(self.base),'--mapping',str(self.mapping),
                         '--output',str(self.base/'output')])
        self.assertEqual(protected.read_text(),'preserve')

    def test_worker_exit_zero_without_outputs_is_failure(self):
        self.write({'case':self.case})
        for name in ('input.png','hint.png'):
            (self.base/name).write_bytes(b'fixture')
        args=['--mode','joint','--data-root',str(self.base),'--mapping',str(self.mapping),
              '--output',str(self.base/'output')]
        with mock.patch.object(runner.subprocess,'run',return_value=mock.Mock(returncode=0)) as run:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(args),1)
        manifest=json.loads((self.base/'output'/'release_run.json').read_text())
        self.assertEqual(manifest['status'],'failed')
        self.assertEqual(manifest['missing_outputs'],['case'])
        self.assertEqual(run.call_args.kwargs['cwd'],self.base/'output')


if __name__=='__main__':
    unittest.main()
