# Tugas Reinforcement Learning — Week 12
**Judul:** Implementasi Q-Learning pada FrozenLake menggunakan Python

---

## Tujuan
1. Memahami konsep Agent, State, Action, dan Reward.
2. Menerapkan algoritma Q-Learning.
3. Mengamati proses pembelajaran agent dalam mencari jalur terbaik.

---

## Langkah 1: Install Library

```python
!pip install gymnasium
```

Library `gymnasium` digunakan sebagai framework environment simulasi untuk Reinforcement Learning. Library ini menyediakan berbagai environment standar, termasuk FrozenLake dan Taxi yang akan digunakan pada tugas ini.

---

## Langkah 2: Import Library

```python
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
```

- `numpy` — digunakan untuk operasi matriks pada Q-Table.
- `gymnasium` — menyediakan environment simulasi RL.
- `random` — digunakan untuk memilih aksi secara acak (exploration).
- `matplotlib.pyplot` — digunakan untuk memvisualisasikan reward per episode.

---

## Langkah 3: Membuat Environment

```python
env = gym.make("FrozenLake-v1", is_slippery=False)

print("Jumlah State :", env.observation_space.n)
print("Jumlah Action :", env.action_space.n)
```

Environment **FrozenLake-v1** mensimulasikan peta beku berukuran 4×4 (16 state). Agent harus berjalan dari titik Start (S) ke Goal (G) tanpa jatuh ke lubang (H). Parameter `is_slippery=False` berarti agent bergerak tepat sesuai perintah, tanpa efek acak akibat permukaan licin. Terdapat 4 aksi yang tersedia: kiri, bawah, kanan, atas.

---

## Langkah 4: Inisialisasi Q-Table

```python
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

print(q_table)
```

Q-Table adalah matriks berukuran `(state_size × action_size)` yang menyimpan nilai Q untuk setiap pasangan state-action. Awalnya semua nilai diisi nol karena agent belum memiliki pengetahuan apapun. Nilai-nilai ini akan diperbarui selama proses training.

---

## Langkah 5: Parameter Q-Learning

```python
alpha = 0.8       # learning rate
gamma = 0.95      # discount factor
epsilon = 1.0     # exploration rate

epsilon_decay = 0.995
min_epsilon = 0.01

episodes = 2000
max_steps = 100
```

Penjelasan setiap parameter:

| Parameter | Nilai | Fungsi |
|---|---|---|
| `alpha` (α) | 0.8 | Learning rate — seberapa cepat agent memperbarui Q-value |
| `gamma` (γ) | 0.95 | Discount factor — seberapa besar agent menghargai reward masa depan |
| `epsilon` | 1.0 | Exploration rate awal — agent mulai dengan 100% eksplorasi acak |
| `epsilon_decay` | 0.995 | Faktor penurunan epsilon setiap episode |
| `min_epsilon` | 0.01 | Batas minimum epsilon agar selalu ada sedikit eksplorasi |
| `episodes` | 2000 | Total episode training |
| `max_steps` | 100 | Maksimum langkah per episode |

---

## Langkah 6: Training Q-Learning

```python
rewards = []

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # Exploration vs Exploitation
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q-Table
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards.append(total_reward)

print("Training selesai!")
```

Ini adalah inti dari algoritma Q-Learning. Setiap episode, agent mulai dari state awal dan mengambil aksi berdasarkan strategi **epsilon-greedy**: jika angka acak lebih kecil dari epsilon, agent bereksplorasi (memilih aksi acak); jika tidak, agent mengeksploitasi pengetahuan yang sudah dimiliki (memilih aksi dengan Q-value tertinggi). Setelah mengambil aksi, Q-Table diperbarui menggunakan rumus Bellman:

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
```

Setiap akhir episode, nilai epsilon diturunkan sehingga seiring waktu agent semakin banyak mengeksploitasi dan lebih sedikit bereksplorasi.

---

## Langkah 7: Menampilkan Q-Table

```python
print("Q-Table Hasil Training")
print(q_table)
```

Setelah training selesai, Q-Table dicetak untuk melihat nilai Q yang telah dipelajari agent untuk setiap kombinasi state dan action. Nilai yang lebih tinggi menunjukkan bahwa aksi tersebut dianggap lebih menguntungkan oleh agent pada state tertentu.

---

## Langkah 8: Visualisasi Reward

```python
plt.figure(figsize=(10,5))
plt.plot(rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
```

Grafik ini menampilkan total reward yang diperoleh agent di setiap episode selama training. Pada awal training, reward cenderung rendah atau nol karena agent masih banyak bereksplorasi secara acak. Seiring bertambahnya episode, grafik reward seharusnya menunjukkan tren naik yang menandakan agent semakin mahir menemukan jalur menuju Goal.

---

## Langkah 9: Testing Agent

```python
env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)

state, info = env.reset()

done = False
total_reward = 0

while not done:

    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, info = env.step(action)

    print(env.render())

    total_reward += reward

    done = terminated or truncated

print("Total Reward:", total_reward)
```

Pada fase testing, agent tidak lagi bereksplorasi — ia selalu memilih aksi dengan Q-value tertinggi (`np.argmax`). Environment dibuat dengan `render_mode="ansi"` agar posisi agent dapat divisualisasikan di terminal pada setiap langkah. Jika training berhasil, agent seharusnya mampu menemukan jalur optimal dari Start ke Goal dan mendapatkan total reward sebesar 1.0.

---

## Pertanyaan & Jawaban

### 1. Apa yang dimaksud dengan State, Action, dan Reward dalam Reinforcement Learning?

- **State (s)** adalah representasi kondisi lingkungan yang sedang dihadapi agent pada suatu waktu tertentu. Pada FrozenLake, state adalah posisi agent saat ini di peta (0–15 untuk grid 4×4).
- **Action (a)** adalah pilihan yang dapat diambil oleh agent dari suatu state. Pada FrozenLake terdapat 4 aksi: bergerak ke kiri (0), bawah (1), kanan (2), dan atas (3).
- **Reward (r)** adalah umpan balik numerik dari environment setelah agent mengambil suatu aksi. Pada FrozenLake, agent mendapatkan reward 1.0 jika berhasil mencapai Goal (G), dan 0 untuk langkah lainnya (termasuk jatuh ke lubang).

### 2. Apa fungsi dari Learning Rate (α)?

Learning Rate (α) mengontrol seberapa besar pembaruan yang dilakukan terhadap nilai Q-Table setiap kali agent menerima informasi baru. Nilai α yang tinggi (mendekati 1) membuat agent lebih agresif dalam memperbarui pengetahuannya, sehingga informasi terbaru lebih mendominasi. Nilai α yang rendah membuat pembaruan lebih lambat dan stabil. Pada tugas ini digunakan `α = 0.8`, artinya 80% dari selisih antara nilai Q lama dan target baru diaplikasikan setiap pembaruan.

### 3. Apa fungsi dari Discount Factor (γ)?

Discount Factor (γ) menentukan seberapa besar agent menghargai reward di masa depan dibandingkan reward yang didapat sekarang. Nilai γ mendekati 1 berarti agent sangat mempertimbangkan reward jangka panjang (berpikir jauh ke depan). Nilai γ mendekati 0 berarti agent lebih mementingkan reward jangka pendek. Pada tugas ini `γ = 0.95`, artinya reward di masa depan masih dihargai sebesar 95% dari nilai aslinya, sehingga agent termotivasi untuk mencari jalur terbaik menuju Goal meskipun butuh banyak langkah.

### 4. Mengapa digunakan metode Exploration dan Exploitation?

Exploration dan Exploitation adalah dua strategi yang saling bertentangan namun sama-sama diperlukan:

- **Exploration** (eksplorasi): agent mencoba aksi-aksi secara acak untuk menemukan kemungkinan jalur atau reward baru yang belum diketahui. Tanpa eksplorasi, agent bisa terjebak pada solusi yang suboptimal.
- **Exploitation** (eksploitasi): agent menggunakan pengetahuan yang sudah dipelajari untuk memilih aksi terbaik berdasarkan Q-Table saat ini.

Keseimbangan keduanya dikelola melalui **epsilon-greedy**: di awal training epsilon tinggi sehingga agent banyak mengeksplorasi, lalu epsilon diturunkan secara bertahap (`epsilon_decay = 0.995`) sehingga agent semakin banyak mengeksploitasi pengetahuannya. Tanpa eksplorasi, agent tidak akan pernah menemukan jalur menuju Goal. Tanpa eksploitasi, agent tidak akan pernah mengoptimalkan strategi yang sudah dipelajari.

### 5. Bagaimana perubahan nilai reward setelah training 2000 episode?

Pada awal training (episode 1–beberapa ratus pertama), reward yang diperoleh agent sebagian besar bernilai 0 karena epsilon masih tinggi dan agent banyak bergerak secara acak — termasuk sering jatuh ke lubang sebelum mencapai Goal. Seiring berjalannya episode, epsilon terus menurun sehingga agent mulai lebih sering mengeksploitasi Q-Table yang sudah mulai terbentuk. Memasuki pertengahan hingga akhir training, frekuensi mendapatkan reward 1.0 meningkat signifikan dan grafik reward menunjukkan tren naik yang jelas. Pada akhir 2000 episode, agent yang berhasil dilatih seharusnya secara konsisten mendapatkan reward 1.0 di setiap episode, menandakan ia sudah menemukan jalur optimal menuju Goal.

---

## Tugas Lanjutan

Modifikasi program menggunakan environment **Taxi-v3**, menampilkan rata-rata reward setiap 100 episode, dan membandingkan hasil training 1000, 2000, dan 5000 episode.

### Kode Lengkap Tugas Lanjutan

```python
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# 1. Setup Environment
# ──────────────────────────────────────────
env = gym.make("Taxi-v3")

state_size  = env.observation_space.n   # 500 state
action_size = env.action_space.n        # 6 aksi

print("Jumlah State :", state_size)
print("Jumlah Action:", action_size)

# ──────────────────────────────────────────
# 2. Fungsi Training Q-Learning
# ──────────────────────────────────────────
def train_q_learning(episodes, alpha=0.8, gamma=0.95,
                     epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    q_table = np.zeros((state_size, action_size))
    epsilon = epsilon_start
    all_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(200):  # max_steps untuk Taxi
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(action)

            # Update Q-Table (Bellman Equation)
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

    return q_table, all_rewards

# ──────────────────────────────────────────
# 3. Fungsi Rata-rata Reward per 100 Episode
# ──────────────────────────────────────────
def avg_reward_per_100(rewards):
    avg = []
    for i in range(0, len(rewards), 100):
        chunk = rewards[i:i+100]
        avg.append(np.mean(chunk))
    return avg

# ──────────────────────────────────────────
# 4. Training dengan 3 Variasi Episode
# ──────────────────────────────────────────
print("\nTraining 1000 episode...")
q1000, rewards_1000 = train_q_learning(1000)

print("Training 2000 episode...")
q2000, rewards_2000 = train_q_learning(2000)

print("Training 5000 episode...")
q5000, rewards_5000 = train_q_learning(5000)

print("Semua training selesai!")

# ──────────────────────────────────────────
# 5. Rata-rata Reward Setiap 100 Episode
# ──────────────────────────────────────────
avg1000 = avg_reward_per_100(rewards_1000)
avg2000 = avg_reward_per_100(rewards_2000)
avg5000 = avg_reward_per_100(rewards_5000)

print("\n--- Rata-rata Reward per 100 Episode ---")

print("\n[1000 Episode]")
for i, val in enumerate(avg1000):
    print(f"  Episode {(i+1)*100:>5}: {val:.2f}")

print("\n[2000 Episode]")
for i, val in enumerate(avg2000):
    print(f"  Episode {(i+1)*100:>5}: {val:.2f}")

print("\n[5000 Episode]")
for i, val in enumerate(avg5000):
    print(f"  Episode {(i+1)*100:>5}: {val:.2f}")

# ──────────────────────────────────────────
# 6. Visualisasi Perbandingan
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Raw reward per episode
axes[0].plot(rewards_1000, alpha=0.5, label="1000 Episode")
axes[0].plot(rewards_2000, alpha=0.5, label="2000 Episode")
axes[0].plot(rewards_5000, alpha=0.5, label="5000 Episode")
axes[0].set_title("Reward per Episode — Taxi-v3")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Rata-rata reward per 100 episode
x1 = [i * 100 for i in range(1, len(avg1000) + 1)]
x2 = [i * 100 for i in range(1, len(avg2000) + 1)]
x5 = [i * 100 for i in range(1, len(avg5000) + 1)]

axes[1].plot(x1, avg1000, marker='o', label="1000 Episode")
axes[1].plot(x2, avg2000, marker='s', label="2000 Episode")
axes[1].plot(x5, avg5000, marker='^', label="5000 Episode")
axes[1].set_title("Rata-rata Reward per 100 Episode — Taxi-v3")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Rata-rata Reward")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────
# 7. Perbandingan Hasil Akhir
# ──────────────────────────────────────────
print("\n--- Perbandingan Rata-rata Reward 100 Episode Terakhir ---")
print(f"  1000 Episode: {np.mean(rewards_1000[-100:]):.2f}")
print(f"  2000 Episode: {np.mean(rewards_2000[-100:]):.2f}")
print(f"  5000 Episode: {np.mean(rewards_5000[-100:]):.2f}")
```

### Penjelasan Tugas Lanjutan

**Environment Taxi-v3** jauh lebih kompleks dari FrozenLake. Terdapat 500 state yang merepresentasikan kombinasi posisi taksi (25 lokasi), posisi penumpang (5 kemungkinan), dan tujuan penumpang (4 lokasi). Tersedia 6 aksi: bergerak ke selatan, utara, timur, barat, mengambil penumpang (pickup), dan menurunkan penumpang (dropoff). Reward yang diberikan: +20 jika berhasil mengantar penumpang, -1 per langkah, dan -10 jika melakukan pickup/dropoff di lokasi yang salah.

**Fungsi `train_q_learning`** dimodularisasi agar dapat dipanggil berulang kali dengan jumlah episode berbeda tanpa menduplikasi kode. Setiap pemanggilan menghasilkan Q-Table baru dan daftar reward per episode yang independen.

**Fungsi `avg_reward_per_100`** membagi seluruh daftar reward menjadi kelompok-kelompok 100 episode lalu menghitung rata-ratanya. Ini berguna untuk melicinkan fluktuasi reward dan melihat tren pembelajaran secara lebih jelas dibandingkan melihat nilai reward mentah per episode.

**Perbandingan 1000, 2000, dan 5000 episode** menunjukkan pengaruh durasi training terhadap kualitas agent. Secara umum, semakin banyak episode yang digunakan, semakin tinggi rata-rata reward 100 episode terakhir — karena agent memiliki lebih banyak waktu untuk mengeksplorasi lingkungan dan menyempurnakan Q-Table-nya.

---

## Output yang Diharapkan

- Q-Table terbentuk dengan nilai yang mencerminkan jalur optimal.
- Grafik reward menunjukkan tren naik seiring bertambahnya episode training.
- Agent mampu menemukan jalur optimal menuju goal secara konsisten.
- Pada tugas lanjutan, training 5000 episode menghasilkan rata-rata reward tertinggi dibandingkan 1000 dan 2000 episode.
