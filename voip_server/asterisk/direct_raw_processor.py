#!/usr/bin/env python3
"""
Michell - 2025-10-01

Direct Raw Processor - Processador direto de dados brutos de arquivos WAV em crescimento

Este script acessa diretamente os dados brutos de um arquivo WAV em crescimento,
ignorando o cabeçalho WAV e tratando o arquivo como um fluxo contínuo de dados.

Uso:
    python direct_raw_processor.py arquivo.wav [diretório_saída] [opções]

"""

import os
import sys
import time
import wave
import struct
import argparse
import numpy as np
from datetime import datetime
from collections import deque

try:
    import audioop
except ImportError:
    audioop = None


class DirectRawProcessor:
    """
    Processador direto de dados brutos de arquivos WAV em crescimento.
    """

    def __init__(self, wav_file, output_dir=None, folga_db=12.0, min_silence_len=1000,
                 keep_silence=250, min_segment_len=2500, max_segment_len=10000,
                 sample_rate=8000, sample_width=2, channels=1, stop_on_eof=False):
        """
        Inicializa o processador de dados brutos.

        Args:
            wav_file (str): Caminho para o arquivo WAV em crescimento
            output_dir (str): Diretório para salvar os segmentos
            folga_db (float): Valor inicial em dB para subtrair do volume médio para definir o limiar de silêncio
            min_silence_len (int): Duração mínima do silêncio em ms para considerar um ponto de corte
            keep_silence (int): Quantidade de silêncio a manter no início e fim dos segmentos em ms
            min_segment_len (int): Duração mínima do segmento em ms; segmentos menores são retidos e
                mesclados com o próximo segmento até atingir este tamanho
            max_segment_len (int): Duração máxima do segmento em ms; nenhum segmento gravado excederá
                este limite — se ultrapassar, o limiar de silêncio é apertado iterativamente até que
                todos os subsegmentos fiquem dentro de max_segment_len
            sample_rate (int): Taxa de amostragem em Hz (padrão: 8000; detectada do cabeçalho WAV se disponível)
            sample_width (int): Largura da amostra em bytes (padrão: 2 = 16 bits; detectada do cabeçalho WAV se disponível)
            channels (int): Número de canais (padrão: 1 = mono; detectado do cabeçalho WAV se disponível)
            stop_on_eof (bool): Se True, encerra ao atingir o fim do arquivo (útil para testes em arquivos já gravados)
        """
        self.wav_file = wav_file
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(wav_file))
        self.folga_db = float(folga_db)
        self.current_folga_db = float(folga_db)
        self.min_folga_db = 0.0
        self.folga_step_db = 2.0
        self.folga_relax_db = 1.0

        self.min_silence_len = int(min_silence_len)
        self.keep_silence = int(keep_silence)
        self.min_segment_len = int(min_segment_len)
        self.max_segment_len = int(max_segment_len)
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.stop_on_eof = stop_on_eof

        if self.max_segment_len > 0 and self.min_segment_len > self.max_segment_len:
            raise ValueError("min_segment_len não pode ser maior que max_segment_len")

        # Posição atual no arquivo
        self.current_position = 0

        # Tamanho do cabeçalho WAV padrão (44 bytes; atualizado se cabeçalho WAV for detectado)
        self.header_size = 44

        # Verificar se o arquivo existe ou esperar por ele
        self._wait_for_file()

        # Tentar detectar o formato real a partir do cabeçalho WAV (se presente)
        self._detect_wav_format()

        # Bytes por segundo e tamanho do chunk em bytes (100ms de áudio)
        self.bytes_per_second = self.sample_rate * self.sample_width * self.channels
        self.chunk_size = int(self.bytes_per_second * 0.1)

        # Número de chunks para considerar silêncio
        self.min_silence_chunks = max(1, int(self.min_silence_len / 100))

        # Número de chunks de silêncio a manter
        self.keep_silence_chunks = max(0, int(self.keep_silence / 100))

        # Buffer circular para armazenar os dados de áudio recentes (últimos 100s)
        self.audio_buffer = deque(maxlen=1000)

        # Segmento curto retido aguardando o próximo segmento para atingir min_segment_len
        self.kept_segment = []

        # Contador de segmentos salvos e de chunks processados
        self.segment_count = 0
        self.chunks_processed = 0

        # Histórico de volume para cálculo dinâmico do limiar de silêncio
        self.volume_history = deque(maxlen=50)

        # Limiar de silêncio dinâmico
        self.silence_threshold = None

        # Criar o diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Processador direto iniciado para {wav_file}")
        print(f"Segmentos serão salvos em {self.output_dir}")
        print(
            f"Configuração: folga_db={self.folga_db}, min_silence_len={self.min_silence_len}ms, "
            f"keep_silence={self.keep_silence}ms, min_segment_len={self.min_segment_len}ms, "
            f"max_segment_len={self.max_segment_len}ms"
        )
        print(f"Formato de áudio: {self.channels} canais, {self.sample_width*8} bits, {self.sample_rate} Hz")

    def _wait_for_file(self):
        """Espera até que o arquivo exista."""
        while not os.path.exists(self.wav_file):
            print(f"Aguardando a criação do arquivo {self.wav_file}...")
            time.sleep(1)

    def _detect_wav_format(self):
        """Detecta taxa de amostragem, canais, largura da amostra e offset de dados no cabeçalho WAV."""
        try:
            if os.path.getsize(self.wav_file) < 44:
                return
            with open(self.wav_file, 'rb') as f:
                header = f.read(1024)
            if len(header) >= 44 and header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                pos = 12
                while pos + 8 <= len(header):
                    chunk_id = header[pos:pos + 4]
                    chunk_len = struct.unpack('<I', header[pos + 4:pos + 8])[0]
                    if chunk_id == b'fmt ' and pos + 24 <= len(header):
                        _, ch, sr, _, _, bits = struct.unpack('<HHIIHH', header[pos + 8:pos + 24])
                        if ch > 0 and sr > 0 and bits in (8, 16, 24, 32):
                            self.channels = ch
                            self.sample_rate = sr
                            self.sample_width = bits // 8
                    elif chunk_id == b'data':
                        self.header_size = pos + 8
                        break
                    pos += 8 + chunk_len + (chunk_len % 2)
        except Exception as e:
            print(f"Aviso ao detectar formato WAV (usando padrão): {e}")

    def _read_new_data(self):
        """Lê novos dados do arquivo em crescimento."""
        try:
            with open(self.wav_file, 'rb') as f:
                f.seek(self.current_position)
                chunk = f.read(self.chunk_size)
                # Garantir alinhamento ao frame de áudio
                frame_size = self.sample_width * self.channels
                if frame_size > 0 and len(chunk) % frame_size != 0:
                    aligned_len = len(chunk) - (len(chunk) % frame_size)
                    chunk = chunk[:aligned_len]
                    self.current_position += aligned_len
                else:
                    self.current_position = f.tell()
                return chunk
        except Exception as e:
            print(f"Erro ao ler novos dados: {e}")
            return b''

    def _compute_rms(self, chunk):
        """Calcula o RMS de um chunk de áudio PCM."""
        if not chunk:
            return 0.0
        if audioop is not None:
            return float(audioop.rms(chunk, self.sample_width))
        if self.sample_width == 2:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float64)
        elif self.sample_width == 1:
            samples = np.frombuffer(chunk, dtype=np.uint8).astype(np.float64) - 128.0
        else:
            samples = np.frombuffer(chunk, dtype=np.int32).astype(np.float64)
        return float(np.sqrt(np.mean(samples ** 2))) if samples.size > 0 else 0.0

    def _segment_duration_ms(self, segment):
        """Retorna a duração de um segmento (lista de chunks) em milissegundos."""
        if not segment or self.bytes_per_second <= 0:
            return 0.0
        total_bytes = sum(len(c) for c in segment)
        return (total_bytes / self.bytes_per_second) * 1000.0

    def _calculate_silence_threshold(self, folga_db=None):
        """Calcula o limiar de silêncio dinâmico com base no volume médio."""
        if not self.volume_history:
            return 500.0

        folga = self.current_folga_db if folga_db is None else folga_db
        avg_rms = sum(self.volume_history) / len(self.volume_history)

        if avg_rms > 0:
            avg_db = 20 * np.log10(avg_rms)
            threshold_db = avg_db - folga
            threshold_rms = 10 ** (threshold_db / 20)

            print(
                f"Volume médio: {avg_db:.2f} dB | folga_db={folga:.1f} dB | "
                f"Limiar de silêncio: {threshold_db:.2f} dB ({threshold_rms:.2f} RMS)"
            )
            return float(threshold_rms)
        return 500.0

    def _tighten_silence_threshold(self):
        """Reduz a tolerância ao silêncio (diminui folga_db / eleva limiar RMS) para gerar segmentos menores."""
        if self.current_folga_db > self.min_folga_db:
            self.current_folga_db = max(self.min_folga_db, self.current_folga_db - self.folga_step_db)
            self.silence_threshold = self._calculate_silence_threshold(self.current_folga_db)
            print(
                f"Segmento excedeu max_segment_len ({self.max_segment_len}ms): "
                f"reduzindo tolerância ao silêncio (folga_db -> {self.current_folga_db:.1f} dB)"
            )

    def _relax_silence_threshold(self):
        """Relaxa gradualmente folga_db de volta ao valor configurado quando segmentos ficam curtos demais."""
        if self.current_folga_db < self.folga_db:
            self.current_folga_db = min(self.folga_db, self.current_folga_db + self.folga_relax_db)
            self.silence_threshold = self._calculate_silence_threshold(self.current_folga_db)
            print(
                f"Segmento curto retido: relaxando tolerância ao silêncio "
                f"(folga_db -> {self.current_folga_db:.1f} dB)"
            )

    def _is_silence(self, chunk):
        """Verifica se um chunk de áudio é silêncio."""
        if not chunk:
            return True

        rms = self._compute_rms(chunk)
        self.volume_history.append(rms)
        self.chunks_processed += 1

        # Recalcular o limiar de silêncio a cada 20 chunks (2 segundos)
        if len(self.volume_history) >= 20 and self.chunks_processed % 20 == 0:
            self.silence_threshold = self._calculate_silence_threshold()

        if self.silence_threshold is None:
            self.silence_threshold = 500.0

        return rms < self.silence_threshold

    def _split_chunks_by_threshold(self, segment, rms_vals, threshold_rms, min_sil_chunks):
        """Divide um segmento em subsegmentos usando um limiar RMS e duração mínima de silêncio."""
        pieces = []
        current = []
        pending_sil = []
        sil_count = 0
        in_sil = True
        keep = min(self.keep_silence_chunks, max(0, min_sil_chunks // 2))

        for i, (chunk, rms) in enumerate(zip(segment, rms_vals)):
            if rms < threshold_rms:
                sil_count += 1
                if not in_sil:
                    pending_sil.append(chunk)
                    if sil_count >= min_sil_chunks:
                        in_sil = True
                        if keep > 0:
                            current.extend(pending_sil[:keep])
                        pending_sil = []
                        if current:
                            pieces.append(current)
                        current = []
            else:
                if in_sil:
                    in_sil = False
                    if keep > 0 and sil_count > 0:
                        lead = min(sil_count, keep)
                        current.extend(segment[max(0, i - lead):i])
                elif pending_sil:
                    current.extend(pending_sil)
                    pending_sil = []
                sil_count = 0
                current.append(chunk)

        if pending_sil and keep > 0:
            current.extend(pending_sil[:keep])
        if current:
            pieces.append(current)

        return pieces

    def _split_oversized_segment(self, segment, _depth=0):
        """
        Garante que nenhum subsegmento exceda max_segment_len.
        Reduz iterativamente a tolerância ao silêncio (diminuindo folga_db e min_silence)
        e, se algum pedaço ainda ficar maior que max_segment_len, corta recursivamente
        até todos os pedaços respeitarem max_segment_len.
        """
        if not segment or self.max_segment_len <= 0:
            return [segment] if segment else []

        if self._segment_duration_ms(segment) <= self.max_segment_len:
            return [segment]

        rms_vals = [self._compute_rms(c) for c in segment]
        avg_rms = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0

        # Tentar dividir apertando iterativamente folga_db (menos tolerância ao silêncio)
        # e, se necessário, reduzindo a duração mínima de silêncio exigida para o corte.
        if avg_rms > 0 and _depth < 12:
            avg_db = 20 * np.log10(avg_rms)
            start_folga = self.current_folga_db - self.folga_step_db * (_depth + 1)
            folga_candidates = []
            f = start_folga
            while f >= -6.0:
                folga_candidates.append(f)
                f -= self.folga_step_db

            min_sil_candidates = sorted(
                set([
                    self.min_silence_chunks,
                    max(1, self.min_silence_chunks // 2),
                    max(1, self.min_silence_chunks // 3),
                    2,
                    1,
                ]),
                reverse=True,
            )

            for min_sil in min_sil_candidates:
                for trial_folga in folga_candidates:
                    threshold_rms = 10 ** ((avg_db - trial_folga) / 20)
                    pieces = self._split_chunks_by_threshold(segment, rms_vals, threshold_rms, min_sil)
                    if len(pieces) >= 2:
                        # Verificar se houve progresso real (algum corte reduziu o maior pedaço)
                        max_piece_ms = max(self._segment_duration_ms(p) for p in pieces)
                        if max_piece_ms < self._segment_duration_ms(segment):
                            result = []
                            for p in pieces:
                                if self._segment_duration_ms(p) > self.max_segment_len:
                                    result.extend(self._split_oversized_segment(p, _depth=_depth + 1))
                                else:
                                    result.append(p)
                            return result

        # Fallback garantido: cortar no chunk de menor RMS dentro da janela válida <= max_segment_len
        # (evita cortar no meio de um pico de fala e garante limite rígido <= max_segment_len).
        max_chunks = max(1, int(self.max_segment_len / 100))
        min_chunks = max(1, int(self.min_segment_len / 100))

        # Ajustar max_chunks pela duração real em bytes
        while max_chunks > 1 and self._segment_duration_ms(segment[:max_chunks]) > self.max_segment_len:
            max_chunks -= 1

        search_start = min(min_chunks, max(1, max_chunks - 1))
        search_end = min(max_chunks, len(segment) - 1)

        if search_start >= search_end:
            cut_idx = max(1, min(max_chunks, len(segment) - 1))
        else:
            # Escolher o ponto de menor energia RMS na janela [search_start, search_end]
            window = rms_vals[search_start:search_end + 1]
            cut_idx = search_start + int(np.argmin(window))

        left = segment[:cut_idx]
        right = segment[cut_idx:]

        result = []
        if left:
            if self._segment_duration_ms(left) > self.max_segment_len:
                result.extend(self._split_oversized_segment(left, _depth=_depth + 1))
            else:
                result.append(left)
        if right:
            if self._segment_duration_ms(right) > self.max_segment_len:
                result.extend(self._split_oversized_segment(right, _depth=_depth + 1))
            else:
                result.append(right)
        return result

    def _save_segment(self, segment):
        """Salva um segmento de áudio como arquivo WAV."""
        if not segment:
            return None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        segment_filename = f"segment_{timestamp}_{self.segment_count}.wav"
        segment_path = os.path.join(self.output_dir, segment_filename)
        duration_s = self._segment_duration_ms(segment) / 1000.0

        with wave.open(segment_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(segment))

        print(f"Segmento salvo: {segment_path} ({duration_s:.2f}s)")
        self.segment_count += 1

        return segment_path

    def _handle_completed_segment(self, segment, allow_split=True):
        """
        Aplica as regras de min_segment_len e max_segment_len:
        - Se o segmento (ou kept_segment + segment) exceder max_segment_len, aperta o limiar
          e recorta iterativamente até todos os pedaços ficarem <= max_segment_len.
        - Se (kept_segment + segment) >= min_segment_len (e <= max_segment_len), salva e limpa kept_segment.
        - Caso contrário, retém o combinado em kept_segment para unir ao próximo segmento.
        """
        if not segment:
            return

        # Ignorar ruído/silêncio puro muito pequeno quando não há segmento retido
        if not self.kept_segment and len(segment) <= self.keep_silence_chunks * 2:
            return

        combined = self.kept_segment + segment
        combined_ms = self._segment_duration_ms(combined)

        if allow_split and self.max_segment_len > 0 and combined_ms > self.max_segment_len:
            self._tighten_silence_threshold()
            self.kept_segment = []
            pieces = self._split_oversized_segment(combined)
            for piece in pieces:
                self._handle_completed_segment(piece, allow_split=False)
            return

        if combined_ms >= self.min_segment_len:
            self._save_segment(combined)
            self.kept_segment = []
        else:
            self.kept_segment = combined
            print(
                f"Segmento retido ({combined_ms / 1000.0:.2f}s < "
                f"{self.min_segment_len / 1000.0:.2f}s); aguardando próximo segmento"
            )
            self._relax_silence_threshold()

    def process(self):
        """
        Processa o arquivo em crescimento e o segmenta em pontos de silêncio.
        """
        current_segment = []
        pending_silence = []
        silence_count = 0
        is_in_silence = True

        try:
            # Pular o cabeçalho WAV se o arquivo for grande o suficiente
            if os.path.getsize(self.wav_file) > self.header_size:
                self.current_position = self.header_size
                print(f"Pulando cabeçalho WAV ({self.header_size} bytes)")

            while True:
                if not os.path.exists(self.wav_file):
                    if self.stop_on_eof:
                        break
                    print(f"Arquivo {self.wav_file} não encontrado. Aguardando...")
                    time.sleep(1)
                    continue

                chunk = self._read_new_data()

                if not chunk:
                    if self.stop_on_eof:
                        break
                    time.sleep(0.1)
                    continue

                self.audio_buffer.append(chunk)
                is_sil = self._is_silence(chunk)

                if is_sil:
                    silence_count += 1
                    if not is_in_silence:
                        pending_silence.append(chunk)
                        if silence_count >= self.min_silence_chunks:
                            is_in_silence = True
                            if self.keep_silence_chunks > 0:
                                current_segment.extend(pending_silence[:self.keep_silence_chunks])
                            pending_silence = []

                            self._handle_completed_segment(current_segment)
                            current_segment = []
                else:
                    if is_in_silence:
                        is_in_silence = False
                        if self.keep_silence_chunks > 0 and silence_count > 0:
                            silence_to_keep = min(silence_count, self.keep_silence_chunks)
                            buf_list = list(self.audio_buffer)
                            start_idx = max(0, len(buf_list) - 1 - silence_to_keep)
                            current_segment.extend(buf_list[start_idx:len(buf_list) - 1])
                    elif pending_silence:
                        # Preservar pausas curtas dentro da fala para manter continuidade do áudio
                        current_segment.extend(pending_silence)
                        pending_silence = []

                    silence_count = 0
                    current_segment.append(chunk)

                # Garantir em tempo real que o buffer ativo não ultrapasse max_segment_len
                if self.max_segment_len > 0 and current_segment:
                    active_buffer = self.kept_segment + current_segment + pending_silence
                    if self._segment_duration_ms(active_buffer) > self.max_segment_len:
                        self._tighten_silence_threshold()
                        pieces = self._split_oversized_segment(active_buffer)
                        self.kept_segment = []
                        pending_silence = []
                        if len(pieces) > 1:
                            # Emitir/processar todos os pedaços anteriores e manter o último ativo no fluxo
                            for piece in pieces[:-1]:
                                self._handle_completed_segment(piece, allow_split=False)
                            current_segment = pieces[-1]
                        else:
                            current_segment = pieces[0] if pieces else []

        except KeyboardInterrupt:
            print("\nProcessamento interrompido pelo usuário.")
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
        finally:
            if pending_silence and self.keep_silence_chunks > 0:
                current_segment.extend(pending_silence[:self.keep_silence_chunks])
            final_segment = self.kept_segment + current_segment
            self.kept_segment = []
            if final_segment and len(final_segment) > self.keep_silence_chunks * 2:
                if self.max_segment_len > 0 and self._segment_duration_ms(final_segment) > self.max_segment_len:
                    for piece in self._split_oversized_segment(final_segment):
                        if piece:
                            self._save_segment(piece)
                else:
                    self._save_segment(final_segment)


def main():
    parser = argparse.ArgumentParser(
        description='Processa diretamente arquivos WAV em crescimento, segmentando por silêncio.'
    )
    parser.add_argument('wav_file', help='Caminho para o arquivo WAV em crescimento')
    parser.add_argument('output_dir', nargs='?', default=None, help='Diretório para salvar os segmentos')
    parser.add_argument('--folga-db', '-f', type=float, default=12.0,
                        help='Valor em dB para subtrair do volume médio (padrão: 12.0)')
    parser.add_argument('--min-silence', '-m', type=int, default=1000,
                        help='Duração mínima do silêncio em ms (padrão: 1000)')
    parser.add_argument('--keep-silence', '-k', type=int, default=250,
                        help='Quantidade de silêncio a manter no início e fim dos segmentos em ms (padrão: 250)')
    parser.add_argument('--min-segment-len', type=int, default=2500,
                        help='Duração mínima do segmento em ms antes de gravar (padrão: 2500)')
    parser.add_argument('--max-segment-len', type=int, default=10000,
                        help='Duração máxima do segmento em ms; segmentos maiores são recortados iterativamente (padrão: 10000)')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Taxa de amostragem padrão em Hz se não houver cabeçalho WAV (padrão: 8000)')
    parser.add_argument('--sample-width', type=int, default=2,
                        help='Largura da amostra padrão em bytes (padrão: 2)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Número padrão de canais (padrão: 1)')
    parser.add_argument('--stop-on-eof', action='store_true',
                        help='Encerrar ao atingir o fim do arquivo (modo de teste em arquivos gravados)')

    args = parser.parse_args()

    processor = DirectRawProcessor(
        wav_file=args.wav_file,
        output_dir=args.output_dir,
        folga_db=args.folga_db,
        min_silence_len=args.min_silence,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
        max_segment_len=args.max_segment_len,
        sample_rate=args.sample_rate,
        sample_width=args.sample_width,
        channels=args.channels,
        stop_on_eof=args.stop_on_eof,
    )
    processor.process()


if __name__ == "__main__":
    main()