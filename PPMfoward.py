import numpy as np

class simulator_1D:

    epsle_0 = 8.854187817e-12  # permittvity of the air
    mu0 = 4 * np.pi * 1e-7  # Magnetic Permeability of the air
    c = 299792458   # speed of light

    def __init__(self):
        # Global Variables
        self.windowNum = 0  # 0 for no window, 6 for Hann window
        self.fmin = 0
        self.fmax = 0
        self.fs = 80e6/3
        self.frequencies = np.array([0.0])
        self.pulse_spectrum = np.array([0.0])
        self.mf_signal_spectrum = np.array([0.0])
        self.window = np.array([0.0])

    def set_model(self, pulse_spectrum, mf_signal_spectrum, frequencies, fs, fmin, fmax, windowNum):
        self.fs = fs
        self.windowNum = windowNum
        self.fmin = fmin
        self.fmax = fmax
        self.pulse_spectrum = pulse_spectrum
        self.mf_signal_spectrum = mf_signal_spectrum
        self.frequencies = frequencies
        self.window = self.mfwindow(frequencies, fmin, fmax, windowNum)
        self.nf = len(frequencies)

    def mfwindow(self, f, fmin, fmax, windowNum):
        f = np.insert(f, 0, 0.0)
        i1 = np.argmin(np.abs(f - fmin))
        i2 = np.argmin(np.abs(f - fmax))
        window_size = i2 - i1
        kaiser_window = np.kaiser(window_size, windowNum)
        full_window = np.zeros_like(f, dtype='float64')
        full_window[i1:i2] = kaiser_window
        return full_window
    
    def __runsim__(self, epsion, lossTangent, thickness, dBnoise = -100):
        nlayer = len(thickness)

        # reflection index between each layers
        def __get_reflectidx__(fs, eps, thickness):
            travel_times = np.zeros(nlayer)
            reflection_indices = np.zeros(nlayer, dtype=int)
            for i in range(nlayer):
                travel_times[i] = 2 * thickness[i] / (self.c / np.sqrt(eps[i]))
                reflection_indices[i] = round(np.sum(travel_times) * fs)
            return reflection_indices

        ridx = __get_reflectidx__(self.fs, epsion, thickness)

        epsion = np.concatenate(([1.0], epsion))
        lossTangent = np.concatenate(([0.0], lossTangent))
        epsion = epsion - 1j * lossTangent * epsion

        # Precompute frequency-dependent values
        omega = 2 * np.pi * self.frequencies[:, np.newaxis]
        k = omega * np.sqrt(epsion * self.mu0 * self.epsle_0)
        kz = np.sqrt(k**2)

        # compute p and r
        p = epsion[1:] * kz[:, :-1] / (epsion[:-1] * kz[:, 1:])
        r = (1 - p) / (1 + p)

        # Precomputed exponential terms
        exp_pos = np.exp(1j * kz[:, 1:-1] * thickness)
        exp_neg = np.exp(-1j * kz[:, 1:-1] * thickness)

        # initialization VV
        VV = np.eye(2, dtype=complex)[np.newaxis, :, :].repeat(self.nf, axis=0)

        # Calculate the propagation matrix of the middle layer
        v = 0.5 * (1 + p[:, :-1])
        V11 = v * exp_pos
        V12 = v * r[:, :-1] * exp_pos
        V21 = v * r[:, :-1] * exp_neg
        V22 = v * exp_neg

        # Computing cumulative effects using matrix multiplication
        for n in range(nlayer):
            V = np.array([[V11[:, n], V12[:, n]], [V21[:, n], V22[:, n]]]).transpose(2, 0, 1)
            VV = np.matmul(V, VV)

        # Calculate the propagation matrix of the last layer
        v1 = 0.5 * (1 + p[:, -1])
        Vtn11 = v1 * 1
        Vtn12 = v1 * r[:, -1]
        Vtn21 = Vtn12
        Vtn22 = Vtn11
        Vtn = np.array([[Vtn11, Vtn12], [Vtn21, Vtn22]]).transpose(2, 0, 1)

        # Final matrix multiplication
        VV = np.matmul(Vtn, VV)

        # Calculate the reflection coefficient
        R_values = -VV[:, 0, 1] / VV[:, 0, 0]
        R_values = np.insert(R_values, 0 , 0.0) # The reflection coefficient at 0 frequency is 0

        # Add noise in the amplitute and phase term of reflectivity
        if dBnoise > -100:
            Prx = np.max(R_values) ** 2
            Pnoise = Prx * 10 ** (dBnoise / 10) * np.sqrt(len(R_values))
            noise = np.sqrt(Pnoise) * (2 * np.random.rand(len(R_values)) - 1 + 1j * (2 * np.random.rand(len(R_values)) - 1))
            R_values = R_values + noise

        # Calculate reflected waveform using range compression
        reflection_wave_spectrum = self.pulse_spectrum * R_values
        comp_wave = self.__range_compress__(reflection_wave_spectrum, self.mf_signal_spectrum, self.window) 
      
        ridx = np.insert(ridx, 0, 0)  # add surface reflection idex
        return ridx + len(np.abs(comp_wave))//2 - 1, comp_wave
 

    def __range_compress__(self, data, mfilter, window):
        out = data * mfilter * window
        out = np.concatenate([out, np.zeros(self.nf, dtype=complex)])
        return np.fft.fftshift(np.fft.ifft(out))
