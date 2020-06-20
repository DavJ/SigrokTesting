import subprocess
import numpy


class SampleRate:

    supported_rates = [(20, 'kHz'), (25, 'kHz'), (50, 'kHz'), (100, 'kHz'),
                       (200, 'kHz'), (250, 'kHz'), (500, 'kHz'), (1, 'MHz'),
                       (2, 'MHz'), (5, 'MHz'), (10, 'MHz'), (20, 'MHz'),
                       (25, 'MHz'), (50, 'MHz'), (100, 'MHz'), (125, 'MHz')]

    def __init__(self, rate, unit):
        if (rate, unit) in self.supported_rates:
            self.rate = rate
            self.unit = unit
        else:
            raise NotImplementedError(f'unsupported sample rate {rate}{unit}')

    @property
    def cli_rate(self):
        return f'{self.rate}{self.unit}'

    @property
    def value(self):
        def unsupported_unit(unit):
            raise NotImplementedError(f'unsupported unit {unit}')

        return self.rate * (
            1e+3 if self.unit == 'kHz' else 1e+6
            if self.unit == 'MHz' else unsupported_unit(self.unit))


SAMPLE_RATE = SampleRate(10, 'MHz')
DEVICE = 'hantek-dso'
BUFFER_SIZE = 32768
NUMBER_OF_CHANNELS = 2


class Hantek():
    def __init__(self,
                 device=DEVICE,
                 sample_rate=SAMPLE_RATE,
                 number_of_channels=NUMBER_OF_CHANNELS,
                 buffer_size=BUFFER_SIZE):
        self.device = device
        self.sample_rate = sample_rate
        self.number_of_channels = number_of_channels
        self.buffer_size = buffer_size
        self.samples = None
        self._opened = False
        self._process = None

    @property
    def sigrok_command(self):
        return (
            f'sigrok-cli -d {self.device} --config "samplerate={self.sample_rate.cli_rate}:'
            f'buffersize={self.buffer_size}" -O analog --continuous')

    def open(self):
        try:
            self._process = subprocess.Popen(
                self.sigrok_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            self._opened = True
        except Exception:
            self._opened = False
            raise ConnectionError(f'cannot open device {self.device}')

        return self._opened

    def close(self):
        self._process.terminate()
        self._opened = False

    def acquire(self, number_of_samples):
        def parse_line(line):
            split_line = line.split()
            if split_line[0][:2] == b'CH':
                channel = int(split_line[0][2:-1]) - 1
                value = float(split_line[1])

            else:
                return

            return channel, value

        if self._opened:
            self.samples = numpy.zeros(
                shape=(self.number_of_channels, number_of_samples))

            counters = [0 for counter in range(self.number_of_channels)]
            while any(counter < number_of_samples for counter in counters):

                line = self._process.stdout.readline()
                try:
                    (channel, value) = parse_line(line)
                    self.samples[channel, counters[channel]] = value
                    counters[channel] += 1

                except:
                    pass
            return self.samples

    def sample_for_time(self, duration):
        number_of_samples = self.sample_rate.value * duration
        return self.acquire(int(number_of_samples))


device = Hantek()
device.open()
samples = device.sample_for_time(duration=0.001)
device.close()
