"""
Shared plot-output helpers for exp2 plotting entry points.

The plotting implementations save PNG files. These helpers route those PNGs
into a png/ subfolder, mirror every figure as PDF into a pdf/ subfolder, and
force a consistent export DPI.
"""
import os
from contextlib import contextmanager

import matplotlib.figure
import matplotlib.pyplot as plt


EXPORT_DPI = 400


def split_output_roots(base_root):
    return os.path.join(base_root, 'png'), os.path.join(base_root, 'pdf')


def expected_pdf_files(expected_png_files):
    return {
        os.path.splitext(name)[0] + '.pdf'
        for name in expected_png_files
    }


def verify_expected(outdir, expected):
    missing = sorted(name for name in expected if not os.path.exists(os.path.join(outdir, name)))
    if missing:
        raise RuntimeError(
            'Missing expected plots in {0}:\n  {1}'.format(outdir, '\n  '.join(missing))
        )
    print(f'\nVerified {len(expected)} expected plots in {outdir}')


@contextmanager
def mirror_png_to_pdf(png_root, pdf_root, dpi=EXPORT_DPI):
    """
    Mirror every PNG save below png_root to a same-relative-path PDF below
    pdf_root, while forcing the requested DPI for both outputs.
    """
    orig_fig_savefig = matplotlib.figure.Figure.savefig
    orig_plt_savefig = plt.savefig

    png_root_abs = os.path.abspath(png_root)

    def _save_with_pdf(save_func, owner, fname, *args, **kwargs):
        if isinstance(fname, (str, os.PathLike)):
            path = os.fspath(fname)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.png':
                png_path = os.path.abspath(path)
                kwargs['dpi'] = dpi
                os.makedirs(os.path.dirname(png_path), exist_ok=True)

                result = save_func(owner, png_path, *args, **kwargs)

                try:
                    rel = os.path.relpath(png_path, png_root_abs)
                    if not rel.startswith('..'):
                        pdf_path = os.path.join(pdf_root, os.path.splitext(rel)[0] + '.pdf')
                        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                        save_func(owner, pdf_path, *args, **kwargs)
                except ValueError:
                    pass
                return result
        return save_func(owner, fname, *args, **kwargs)

    def fig_savefig(self, fname, *args, **kwargs):
        return _save_with_pdf(orig_fig_savefig, self, fname, *args, **kwargs)

    def pyplot_savefig(fname, *args, **kwargs):
        if isinstance(fname, (str, os.PathLike)):
            path = os.fspath(fname)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.png':
                png_path = os.path.abspath(path)
                kwargs['dpi'] = dpi
                os.makedirs(os.path.dirname(png_path), exist_ok=True)

                result = orig_plt_savefig(png_path, *args, **kwargs)

                try:
                    rel = os.path.relpath(png_path, png_root_abs)
                    if not rel.startswith('..'):
                        pdf_path = os.path.join(pdf_root, os.path.splitext(rel)[0] + '.pdf')
                        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                        orig_plt_savefig(pdf_path, *args, **kwargs)
                except ValueError:
                    pass
                return result
        return orig_plt_savefig(fname, *args, **kwargs)

    matplotlib.figure.Figure.savefig = fig_savefig
    plt.savefig = pyplot_savefig
    try:
        yield
    finally:
        matplotlib.figure.Figure.savefig = orig_fig_savefig
        plt.savefig = orig_plt_savefig
