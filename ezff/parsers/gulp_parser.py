"""Interface to GULP, the General Utility Lattice Program."""

from pathlib import PurePath

from .base_parser import BaseParser


class GulpParser(BaseParser):
    """Instance of BaseParser with some extra extraction functions typically used when processing .out files."""

    def __init__(self, filepath: PurePath):
        super().__init__(filepath)
        self.__extractors = {
            'energy': None,

            'atoms': None,
            'cell': None,

            'bulk_modulus': self.bulk_modulus,
            'young_modulus': self.young_modulus,
            'elastic_modulus': self.elastic_modulus,
            'shear_modulus': self.shear_modulus,

            'phonon_disp': None,
            'phonon_gamma': self.phonon_gamma,
            'phonon_kpoints': self.phonon_kpoints,
        }
        
    def elastic_modulus(self, content: str) -> List[np.ndarray] | None:
        pattern = r"Elastic Constant Matrix:\s*\(Units=GPa\)\s*\n-+\s*\n\s*Indices.*?\n-+\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n){6})\s*-+"
        matches = re.findall(pattern, content, re.DOTALL)

        moduli_array = []
        for match in matches:
            matrix_lines = match.strip().split('\n')
            moduli = np.zeros((6, 6))
            for i, line in enumerate(matrix_lines):
                values = re.findall(r'[-\d.]+|\*+', line)[1:]
                for j, val in enumerate(values):
                    try:
                        moduli[i, j] = float(val)
                    except ValueError:
                        moduli[i, j] = 1e6
            moduli_array.append(moduli)
        if not moduli_array:
            return [np.full((6, 6), 1e6)]

        return moduli_array

    def young_modulus(self, content: str) -> List[List[float]]:
        pattern = r"Youngs\s+Moduli\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
        matches = re.findall(pattern, content)

        def parse_value(value: str) -> float:
            try:
                return float(value)
            except ValueError:
                return 1e6

        youngs_moduli = [[parse_value(val) for val in match] for match in matches]

        return youngs_moduli or [[1e6]]

    def bulk_modulus(self, content: str) -> List[List[float]]:
        pattern = r"Bulk\s+Modulus\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
        matches = re.findall(pattern, content)

        def parse_value(value: str) -> float:
            try:
                return float(value)
            except ValueError:
                return 1e6

        bulk_moduli = [[parse_value(val) for val in match] for match in matches]

        return bulk_moduli or [[1e6]]

    def shear_modulus(self, content: str) -> List[List[float]]:
        pattern = r"Shear\s+Modulus\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
        matches = re.findall(pattern, content)

        def parse_value(value: str) -> float:
            try:
                return float(value)
            except ValueError:
                return 1e6

        shear_moduli = [[parse_value(val) for val in match] for match in matches]

        return shear_moduli or [[1e6]]

    def phonon_gamma(self, content: str, n: int = 100) -> List[List[float]]:
        pattern = r"Frequencies \(cm-1\) \[NB: Negative implies an imaginary mode\]:\s*\n\s*((?:[-\d.\s]+\n)+)"
        frequency_blocks = re.findall(pattern, content, re.MULTILINE)

        all_frequencies = []

        for block in frequency_blocks:
            frequencies = re.findall(r'-?\d+\.?\d*', block)
            frequencies = [float(freq) for freq in frequencies]

            if len(frequencies) < n:
                frequencies.extend([-1000] * (n - len(frequencies)))

            frequencies = frequencies[:n]
            all_frequencies.append(frequencies)

        if not all_frequencies:
            return [[-1000] * n]

        return all_frequencies

    def phonon_kpoints(self, content: str) -> List[List[float]]:
        pattern = r"K\s+point\s+\d+\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        matches = re.findall(pattern, content)

        def parse_value(value: str) -> float:
            try:
                return float(value)
            except ValueError:
                return 1e6

        k_points = []
        for match in matches:
            try:
                point = [parse_value(val) for val in match]
                k_points.append(point)
            except Exception:
                k_points.append([1e6, 1e6, 1e6])

        return k_points if k_points else [[1e6, 1e6, 1e6]]

class GulpParserFFTuning(BaseParser):
    def __init__(self, filepath: PurePath):
        super().__init__(filepath)
        self.__extractors = {
            "final summ of squaares": get_final_sum,
        }

    def get_final_sum(content):
        final_sum = 10**10
        for line in content:
            if "Final sum of squares" in line:
                try:
                    final_sum = float(line.split('=')[-1].strip())
                except ValueError:
                    final_sum = 10**10
                break
        return final_sum
