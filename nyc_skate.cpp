// ============================================================================
//  NYC SKATE '02  --  a single-file 3D street skating game
//
//  An early-2000s New York block: storefronts, a granite plaza with a
//  fountain, the "Banks" stair set, a waterfront rail, traffic, pedestrians
//  and pigeons.  Ollie, flip, spin, grab, grind and manual your way to a
//  high score.  No external assets: all geometry, textures and text are
//  generated at startup.
//
//  Build
//    macOS : clang++ -std=c++17 -O2 nyc_skate.cpp -framework OpenGL -framework GLUT -o nyc_skate
//    Linux : g++ -std=c++17 -O2 nyc_skate.cpp -lGL -lglut -o nyc_skate      (freeglut)
//    Win   : cl /std:c++17 /O2 /EHsc nyc_skate.cpp freeglut.lib opengl32.lib
//
//  Controls (also shown in game, H toggles the full panel)
//    W / Up        push (ground)          S / Down   brake
//    A D / Left Right   turn (ground) / spin (air) / balance (grind)
//    SPACE         hold to crouch, release to ollie (longer hold = higher)
//    J kickflip  K heelflip  L pop shove-it  U 360 flip  I impossible
//      (tap the same flip again mid-air for doubles / triples)
//    O (hold)      grab: alone = Indy, +W = Nosegrab, +S = Tailgrab
//    M (hold)      manual (W/S keep balance).  Links tricks into combos.
//    Rails/ledges  land on them to grind.  W or S held on contact =
//                  nosegrind / 5-0; arrive sideways = boardslide.
//                  J K L U I while grinding switch grinds (more points).
//    R reset   P pause   H help   ESC quit
// ============================================================================

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#else
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/glut.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef GL_GENERATE_MIPMAP
#define GL_GENERATE_MIPMAP 0x8191
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif

// ----------------------------------------------------------------------------
//  Math
// ----------------------------------------------------------------------------
static const float PI = 3.14159265358979f;

struct V3 {
    float x, y, z;
    V3() : x(0), y(0), z(0) {}
    V3(float a, float b, float c) : x(a), y(b), z(c) {}
    V3 operator+(const V3& o) const { return V3(x + o.x, y + o.y, z + o.z); }
    V3 operator-(const V3& o) const { return V3(x - o.x, y - o.y, z - o.z); }
    V3 operator*(float s) const { return V3(x * s, y * s, z * s); }
    V3 operator-() const { return V3(-x, -y, -z); }
    V3& operator+=(const V3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    V3& operator-=(const V3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    V3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};
static inline float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(const V3& a, const V3& b) {
    return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float len(const V3& a) { return sqrtf(dot(a, a)); }
static inline float len2D(float x, float z) { return sqrtf(x * x + z * z); }
static inline V3 normalize(const V3& a) {
    float l = len(a);
    return l > 1e-6f ? a * (1.0f / l) : V3(0, 0, 0);
}
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline V3 lerpv(const V3& a, const V3& b, float t) { return a + (b - a) * t; }
static inline float wrapPi(float a) {
    while (a > PI) a -= 2 * PI;
    while (a < -PI) a += 2 * PI;
    return a;
}
static inline float approachf(float cur, float tgt, float step) {
    return cur < tgt ? std::min(cur + step, tgt) : std::max(cur - step, tgt);
}
static inline float smooth01(float t) { t = clampf(t, 0, 1); return t * t * (3 - 2 * t); }
static inline float degf(float r) { return r * 180.0f / PI; }
static inline float radf(float d) { return d * PI / 180.0f; }
// Forward vector for a heading.  yaw = 0 faces +Z, positive yaw turns toward +X.
static inline V3 fwdFromYaw(float yaw) { return V3(sinf(yaw), 0, cosf(yaw)); }
static inline float yawFromDir(float x, float z) { return atan2f(x, z); }
// Exponential smoothing factor that is frame-rate independent.
static inline float expK(float rate, float dt) { return 1.0f - expf(-rate * dt); }

struct Mat4 { float m[16]; };
static Mat4 matIdentity() {
    Mat4 r;
    for (int i = 0; i < 16; ++i) r.m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    return r;
}
static Mat4 matMul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static Mat4 matPerspective(float fovyDeg, float aspect, float zn, float zf) {
    Mat4 r;
    for (float& v : r.m) v = 0;
    float f = 1.0f / tanf(radf(fovyDeg) * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1;
    r.m[14] = 2 * zf * zn / (zn - zf);
    return r;
}
static Mat4 matLookAt(const V3& eye, const V3& at, const V3& up) {
    V3 f = normalize(at - eye), s = normalize(cross(f, up)), u = cross(s, f);
    Mat4 r = matIdentity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye); r.m[13] = -dot(u, eye); r.m[14] = dot(f, eye);
    return r;
}
// Projects a world point to window pixels (origin bottom-left).  Returns false if behind camera.
static bool projectPoint(const Mat4& viewProj, const V3& p, int w, int h, float& sx, float& sy) {
    const float* m = viewProj.m;
    float cx = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    float cy = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    float cw = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (cw < 0.1f) return false;
    sx = (cx / cw * 0.5f + 0.5f) * w;
    sy = (cy / cw * 0.5f + 0.5f) * h;
    return true;
}

// ----------------------------------------------------------------------------
//  Random numbers
// ----------------------------------------------------------------------------
static uint32_t g_rng = 0x9E3779B9u;
static inline uint32_t xrand() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
static inline float frand() { return (xrand() & 0xFFFFFF) / 16777216.0f; }
static inline float frange(float a, float b) { return a + (b - a) * frand(); }
static inline int irange(int a, int b) { return a + (int)(xrand() % (uint32_t)(b - a + 1)); }
static inline uint32_t hashu(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// Deterministic hash in [0,1) used for static world variation.
static inline float hash01(int a, int b = 0, int c = 0) {
    uint32_t h = hashu((uint32_t)a * 73856093u ^ hashu((uint32_t)b * 19349663u ^ (uint32_t)c * 83492791u));
    return (h & 0xFFFFFF) / 16777216.0f;
}
// Tileable value noise; `period` is the lattice period in cells.
static float vnoise(float x, float y, int seed, int period) {
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float fx = x - xi, fy = y - yi;
    auto h = [&](int i, int j) {
        i = ((i % period) + period) % period;
        j = ((j % period) + period) % period;
        return hash01(i, j, seed);
    };
    float a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
    fx = fx * fx * (3 - 2 * fx);
    fy = fy * fy * (3 - 2 * fy);
    return lerpf(lerpf(a, b, fx), lerpf(c, d, fx), fy);
}
static float fbm(float u, float v, int seed, int basePeriod, int octaves) {
    float sum = 0, amp = 0.5f, norm = 0;
    int period = basePeriod;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * vnoise(u * period, v * period, seed + o * 31, period);
        norm += amp;
        amp *= 0.5f;
        period *= 2;
    }
    return sum / norm;
}

// ----------------------------------------------------------------------------
//  Input  (GLUT callbacks write here; the simulation reads it)
// ----------------------------------------------------------------------------
enum { KEY_LEFT = 256, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_F1, KEY_COUNT };
static bool g_keys[KEY_COUNT];
static std::vector<int> g_presses;  // edge-triggered presses since last sim step

static void keyEvent(int k, bool down) {
    if (k < 0 || k >= KEY_COUNT) return;
    if (down && !g_keys[k]) g_presses.push_back(k);
    g_keys[k] = down;
}
static inline bool kFwd() { return g_keys['w'] || g_keys[KEY_UP]; }
static inline bool kBack() { return g_keys['s'] || g_keys[KEY_DOWN]; }
static inline bool kLeft() { return g_keys['a'] || g_keys[KEY_LEFT]; }
static inline bool kRight() { return g_keys['d'] || g_keys[KEY_RIGHT]; }

// ----------------------------------------------------------------------------
//  Procedural textures
// ----------------------------------------------------------------------------
static GLuint g_texBrick, g_texConcrete, g_texAsphalt, g_texGranite, g_texParticle;
static GLuint g_texWood;

static GLuint uploadTexture(int w, int h, const std::vector<unsigned char>& px, bool clampEdges) {
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    GLint wrap = clampEdges ? GL_CLAMP_TO_EDGE : GL_REPEAT;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);
    glGetError();  // anisotropy is optional; ignore if unsupported
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return id;
}
static inline void putGrey(std::vector<unsigned char>& px, int i, float v, float a = 1.0f) {
    unsigned char c = (unsigned char)(clampf(v, 0, 1) * 255.0f);
    px[i * 4 + 0] = c; px[i * 4 + 1] = c; px[i * 4 + 2] = c;
    px[i * 4 + 3] = (unsigned char)(clampf(a, 0, 1) * 255.0f);
}

static void createTextures() {
    // Brick: one tile = 8 bricks wide, 16 courses high (mapped to 2.0m x 1.2m).
    {
        const int N = 256;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                int row = y / 16;
                int xs = (x + (row % 2) * 16) % N;
                int bx = xs / 32, lx = xs % 32, ly = y % 16;
                bool mortar = lx < 2 || ly < 2;
                float n = fbm(x / (float)N, y / (float)N, 7, 8, 4);
                float v = mortar ? 0.92f + 0.06f * n
                                 : (0.62f + 0.22f * hash01(bx, row, 3)) * (0.86f + 0.22f * n);
                if (!mortar && hash01(x, y, 9) < 0.04f) v *= 0.85f;
                putGrey(px, y * N + x, v);
            }
        g_texBrick = uploadTexture(N, N, px, false);
    }
    // Concrete sidewalk slab (one slab per repeat), with seams and gum spots.
    {
        const int N = 128;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                float n = fbm(x / (float)N, y / (float)N, 11, 4, 5);
                float v = 0.78f + 0.16f * n + 0.05f * (hash01(x, y, 2) - 0.5f);
                if (x < 2 || y < 2) v = 0.5f;
                putGrey(px, y * N + x, v);
            }
        for (int g = 0; g < 9; ++g) {  // flattened chewing gum: very NYC
            int cx = 6 + (int)(hash01(g, 1, 5) * (N - 12)), cy = 6 + (int)(hash01(g, 2, 5) * (N - 12));
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx)
                    if (dx * dx + dy * dy <= 4) putGrey(px, (cy + dy) * N + (cx + dx), 0.42f + 0.1f * hash01(g, 3));
        }
        g_texConcrete = uploadTexture(N, N, px, false);
    }
    // Asphalt: dark speckled noise with a few tar-sealed cracks.
    {
        const int N = 256;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                float n = fbm(x / (float)N, y / (float)N, 21, 8, 5);
                float v = 0.5f + 0.3f * n + 0.22f * (hash01(x, y, 4) - 0.5f);
                float crack = fabsf(fbm(x / (float)N, y / (float)N, 33, 3, 3) - 0.5f);
                if (crack < 0.008f) v *= 0.55f;
                putGrey(px, y * N + x, v);
            }
        g_texAsphalt = uploadTexture(N, N, px, false);
    }
    // Granite plaza tile (one 1m tile per repeat).
    {
        const int N = 128;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                float s = hash01(x, y, 6);
                float v = 0.74f + 0.1f * fbm(x / (float)N, y / (float)N, 41, 4, 4);
                if (s < 0.12f) v -= 0.22f;
                else if (s > 0.93f) v += 0.12f;
                if (x < 1 || y < 1) v = 0.52f;
                putGrey(px, y * N + x, v);
            }
        g_texGranite = uploadTexture(N, N, px, false);
    }
    // Plywood for ramps.
    {
        const int N = 128;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                float grain = 0.5f + 0.5f * sinf(x * 0.35f + 6.0f * fbm(x / (float)N, y / (float)N, 51, 2, 3));
                float v = 0.78f + 0.12f * grain + 0.06f * hash01(x, y, 8);
                if (y % 64 < 1) v = 0.5f;
                putGrey(px, y * N + x, v);
            }
        g_texWood = uploadTexture(N, N, px, false);
    }
    // Soft round particle sprite.
    {
        const int N = 64;
        std::vector<unsigned char> px(N * N * 4);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                float dx = (x + 0.5f) / N * 2 - 1, dy = (y + 0.5f) / N * 2 - 1;
                float r = sqrtf(dx * dx + dy * dy);
                float a = clampf(1.0f - r, 0, 1);
                putGrey(px, y * N + x, 1.0f, a * a);
            }
        g_texParticle = uploadTexture(N, N, px, true);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ----------------------------------------------------------------------------
//  Immediate-mode drawing helpers
// ----------------------------------------------------------------------------
struct Col {
    float r, g, b;
    Col() : r(1), g(1), b(1) {}
    Col(float a, float b_, float c) : r(a), g(b_), b(c) {}
    Col operator*(float k) const { return Col(r * k, g * k, b * k); }
};
static inline void glc(const Col& c) { glColor3f(c.r, c.g, c.b); }
static inline void vtx(const V3& p) { glVertex3f(p.x, p.y, p.z); }
static inline void nrm(const V3& n) { glNormal3f(n.x, n.y, n.z); }

// Axis-aligned box with world-space UVs (tu = horizontal tile size, tv = vertical tile size).
static void drawBox(float x0, float y0, float z0, float x1, float y1, float z1, float tu = 1, float tv = 1) {
    float iu = 1 / tu, iv = 1 / tv;
    glBegin(GL_QUADS);
    glNormal3f(0, 1, 0);
    glTexCoord2f(x0 * iu, z0 * iu); glVertex3f(x0, y1, z0);
    glTexCoord2f(x0 * iu, z1 * iu); glVertex3f(x0, y1, z1);
    glTexCoord2f(x1 * iu, z1 * iu); glVertex3f(x1, y1, z1);
    glTexCoord2f(x1 * iu, z0 * iu); glVertex3f(x1, y1, z0);
    glNormal3f(0, -1, 0);
    glTexCoord2f(x0 * iu, z0 * iu); glVertex3f(x0, y0, z0);
    glTexCoord2f(x1 * iu, z0 * iu); glVertex3f(x1, y0, z0);
    glTexCoord2f(x1 * iu, z1 * iu); glVertex3f(x1, y0, z1);
    glTexCoord2f(x0 * iu, z1 * iu); glVertex3f(x0, y0, z1);
    glNormal3f(0, 0, 1);
    glTexCoord2f(x0 * iu, y0 * iv); glVertex3f(x0, y0, z1);
    glTexCoord2f(x1 * iu, y0 * iv); glVertex3f(x1, y0, z1);
    glTexCoord2f(x1 * iu, y1 * iv); glVertex3f(x1, y1, z1);
    glTexCoord2f(x0 * iu, y1 * iv); glVertex3f(x0, y1, z1);
    glNormal3f(0, 0, -1);
    glTexCoord2f(x1 * iu, y0 * iv); glVertex3f(x1, y0, z0);
    glTexCoord2f(x0 * iu, y0 * iv); glVertex3f(x0, y0, z0);
    glTexCoord2f(x0 * iu, y1 * iv); glVertex3f(x0, y1, z0);
    glTexCoord2f(x1 * iu, y1 * iv); glVertex3f(x1, y1, z0);
    glNormal3f(1, 0, 0);
    glTexCoord2f(z1 * iu, y0 * iv); glVertex3f(x1, y0, z1);
    glTexCoord2f(z0 * iu, y0 * iv); glVertex3f(x1, y0, z0);
    glTexCoord2f(z0 * iu, y1 * iv); glVertex3f(x1, y1, z0);
    glTexCoord2f(z1 * iu, y1 * iv); glVertex3f(x1, y1, z1);
    glNormal3f(-1, 0, 0);
    glTexCoord2f(z0 * iu, y0 * iv); glVertex3f(x0, y0, z0);
    glTexCoord2f(z1 * iu, y0 * iv); glVertex3f(x0, y0, z1);
    glTexCoord2f(z1 * iu, y1 * iv); glVertex3f(x0, y1, z1);
    glTexCoord2f(z0 * iu, y1 * iv); glVertex3f(x0, y1, z0);
    glEnd();
}

// Horizontal textured surface, tessellated so per-vertex fog and lighting stay smooth.
static void drawGround(float x0, float z0, float x1, float z1, float y, float tile, float cell = 8.0f) {
    int nx = std::max(1, (int)ceilf((x1 - x0) / cell)), nz = std::max(1, (int)ceilf((z1 - z0) / cell));
    float dx = (x1 - x0) / nx, dz = (z1 - z0) / nz;
    glNormal3f(0, 1, 0);
    glBegin(GL_QUADS);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nz; ++j) {
            float a = x0 + i * dx, b = a + dx, c = z0 + j * dz, d = c + dz;
            glTexCoord2f(a / tile, c / tile); glVertex3f(a, y, c);
            glTexCoord2f(a / tile, d / tile); glVertex3f(a, y, d);
            glTexCoord2f(b / tile, d / tile); glVertex3f(b, y, d);
            glTexCoord2f(b / tile, c / tile); glVertex3f(b, y, c);
        }
    glEnd();
}

// Cylinder (or cone) between two points.
static void drawCyl(const V3& a, const V3& b, float r0, float r1, int seg = 10, bool caps = true) {
    V3 d = b - a;
    float L = len(d);
    if (L < 1e-5f) return;
    V3 ax = d * (1.0f / L);
    V3 t = fabsf(ax.y) < 0.9f ? V3(0, 1, 0) : V3(1, 0, 0);
    V3 u = normalize(cross(ax, t)), v = cross(ax, u);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= seg; ++i) {
        float ang = 2 * PI * i / seg;
        V3 n = u * cosf(ang) + v * sinf(ang);
        nrm(n);
        vtx(a + n * r0);
        vtx(b + n * r1);
    }
    glEnd();
    if (!caps) return;
    for (int e = 0; e < 2; ++e) {
        V3 c = e ? b : a;
        float r = e ? r1 : r0;
        if (r <= 0) continue;
        nrm(e ? ax : -ax);
        glBegin(GL_TRIANGLE_FAN);
        vtx(c);
        for (int i = 0; i <= seg; ++i) {
            float ang = 2 * PI * (e ? i : seg - i) / seg;
            vtx(c + (u * cosf(ang) + v * sinf(ang)) * r);
        }
        glEnd();
    }
}

static GLuint g_sphereList = 0;
static void buildSphereList() {
    g_sphereList = glGenLists(1);
    glNewList(g_sphereList, GL_COMPILE);
    const int ST = 10, SL = 14;
    for (int i = 0; i < ST; ++i) {
        float a0 = PI * (-0.5f + (float)i / ST), a1 = PI * (-0.5f + (float)(i + 1) / ST);
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= SL; ++j) {
            float b = 2 * PI * j / SL, cx = cosf(b), cz = sinf(b);
            glNormal3f(cx * cosf(a1), sinf(a1), cz * cosf(a1));
            glVertex3f(cx * cosf(a1), sinf(a1), cz * cosf(a1));
            glNormal3f(cx * cosf(a0), sinf(a0), cz * cosf(a0));
            glVertex3f(cx * cosf(a0), sinf(a0), cz * cosf(a0));
        }
        glEnd();
    }
    glEndList();
}
static void drawSphere(const V3& c, float rx, float ry, float rz) {
    glPushMatrix();
    glTranslatef(c.x, c.y, c.z);
    glScalef(rx, ry, rz);
    glCallList(g_sphereList);
    glPopMatrix();
}
static void drawSphere(const V3& c, float r) { drawSphere(c, r, r, r); }

// Flat disc on the XZ plane (decals: puddles, shadows, manholes).
static void drawDisc(float cx, float y, float cz, float rx, float rz, int seg = 24) {
    glNormal3f(0, 1, 0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(cx, y, cz);
    for (int i = 0; i <= seg; ++i) {
        float a = 2 * PI * i / seg;
        glVertex3f(cx + cosf(a) * rx, y, cz - sinf(a) * rz);
    }
    glEnd();
}

// ---- Text -------------------------------------------------------------------
static float strokeWidth(const char* s) {
    float w = 0;
    for (const char* p = s; *p; ++p) w += glutStrokeWidth(GLUT_STROKE_ROMAN, *p);
    return w;
}
// Stroke text in the local XY plane, centred on the origin, reading along +X.
// `bold` re-draws with small offsets to thicken strokes (wide lines are not portable).
static void drawStrokeCentered(const char* s, float capH, float maxW, float bold) {
    float sc = capH / 100.0f;
    float w = strokeWidth(s) * sc;
    if (maxW > 0 && w > maxW) { sc *= maxW / w; w = maxW; capH = sc * 100.0f; }
    const int passes = bold > 0 ? 3 : 1;
    for (int ox = 0; ox < passes; ++ox)
        for (int oy = 0; oy < passes; ++oy) {
            glPushMatrix();
            glTranslatef(-w * 0.5f + (ox - 1) * bold * (passes > 1), -capH * 0.5f + (oy - 1) * bold * (passes > 1), 0);
            glScalef(sc, sc, sc);
            for (const char* p = s; *p; ++p) glutStrokeCharacter(GLUT_STROKE_ROMAN, *p);
            glPopMatrix();
        }
}

// ----------------------------------------------------------------------------
//  World layout (metres).  +X east, +Z south (toward the river), +Y up.
// ----------------------------------------------------------------------------
static const float FACADE_Z = -14.0f;   // north storefronts
static const float NCURB_Z = -8.0f;     // north curb
static const float SCURB_Z = 8.0f;      // south curb
static const float PLAZA_Z0 = 14.0f, PLAZA_Z1 = 38.0f;
static const float BULKHEAD_Z = 44.0f;  // waterfront edge
static const float CURB_H = 0.15f;
static const float WATER_Y = -1.1f;
static const float WORLD_X = 55.0f;     // playable half-width
static const float STEP_UP = 0.17f;     // max height the board rolls up (curbs yes, stairs no)
static const float PLAYER_R = 0.32f;
static const float GRAVITY = 18.0f;

enum Zone { ZN_SIDEWALK, ZN_STREET, ZN_PLAZA, ZN_RIVER, ZN_FOUNTAIN, ZN_PIT };
static int zoneAt(float x, float z) {
    if (z >= BULKHEAD_Z) return ZN_RIVER;
    if (x > -3.6f && x < 3.6f && z > 22.4f && z < 29.6f) return ZN_FOUNTAIN;
    if (x > -25.94f && x < -22.0f && z > 9.06f && z < 11.54f) return ZN_PIT;
    if (z >= NCURB_Z && z < SCURB_Z) return ZN_STREET;
    if (z >= PLAZA_Z0 && z < PLAZA_Z1) return ZN_PLAZA;
    return ZN_SIDEWALK;
}
static float baseGround(float x, float z) {
    switch (zoneAt(x, z)) {
        case ZN_STREET: return 0.0f;
        case ZN_RIVER: return -4.0f;
        case ZN_PIT: return -3.0f;
        default: return CURB_H;  // sidewalks, plaza, fountain basin floor
    }
}

// Solids: axis-aligned boxes or ramps.  Used for collision, ground height and (optionally) rendering.
enum SolidStyle { ST_GRANITE, ST_CONCRETE, ST_WOOD, ST_METAL, ST_PAINTED, ST_BRICK };
enum RampAxis { RAMP_NONE, RAMP_PX, RAMP_NX, RAMP_PZ, RAMP_NZ };  // direction the surface rises toward
struct Solid {
    float x0, z0, x1, z1;  // footprint
    float y0;              // bottom (things can pass under raised solids)
    float yLow, yHigh;     // top surface (box: equal)
    int ramp, style;
    Col col;
    bool render, camBlock;
};
static std::vector<Solid> g_solids;

static float solidTop(const Solid& s, float x, float z) {
    if (s.ramp == RAMP_NONE) return s.yHigh;
    float t;
    switch (s.ramp) {
        case RAMP_PX: t = (x - s.x0) / (s.x1 - s.x0); break;
        case RAMP_NX: t = (s.x1 - x) / (s.x1 - s.x0); break;
        case RAMP_PZ: t = (z - s.z0) / (s.z1 - s.z0); break;
        default: t = (s.z1 - z) / (s.z1 - s.z0); break;
    }
    return lerpf(s.yLow, s.yHigh, clampf(t, 0, 1));
}
static void solidSlope(const Solid& s, float& gx, float& gz) {
    gx = gz = 0;
    float dh = s.yHigh - s.yLow;
    switch (s.ramp) {
        case RAMP_PX: gx = dh / (s.x1 - s.x0); break;
        case RAMP_NX: gx = -dh / (s.x1 - s.x0); break;
        case RAMP_PZ: gz = dh / (s.z1 - s.z0); break;
        case RAMP_NZ: gz = -dh / (s.z1 - s.z0); break;
        default: break;
    }
}
static int addBox(float x0, float z0, float x1, float z1, float y0, float y1, int style, Col c,
                  bool render = true, bool camBlock = false) {
    Solid s;
    s.x0 = std::min(x0, x1); s.x1 = std::max(x0, x1);
    s.z0 = std::min(z0, z1); s.z1 = std::max(z0, z1);
    s.y0 = y0; s.yLow = s.yHigh = y1;
    s.ramp = RAMP_NONE; s.style = style; s.col = c;
    s.render = render; s.camBlock = camBlock;
    g_solids.push_back(s);
    return (int)g_solids.size() - 1;
}
static int addRamp(float x0, float z0, float x1, float z1, float yLow, float yHigh, int axis, Col c) {
    int i = addBox(x0, z0, x1, z1, 0.0f, yHigh, ST_WOOD, c);
    g_solids[i].yLow = yLow;
    g_solids[i].ramp = axis;
    return i;
}

// Rails: grindable segments (metal bars, ledge edges, curbs).
enum RailKind { RK_METAL, RK_LEDGE, RK_CURB };
struct Rail {
    V3 a, b;
    int kind;
    const char* name;
    bool drawTube;  // render as a metal pipe with posts
};
static std::vector<Rail> g_rails;
static void addRail(const V3& a, const V3& b, int kind, const char* name, bool drawTube) {
    Rail r; r.a = a; r.b = b; r.kind = kind; r.name = name; r.drawTube = drawTube;
    g_rails.push_back(r);
}
// Parameter of the point on the rail closest to (x,z) in the horizontal plane.
static float railParam(const Rail& r, float x, float z) {
    float dx = r.b.x - r.a.x, dz = r.b.z - r.a.z;
    float l2 = dx * dx + dz * dz;
    if (l2 < 1e-6f) return 0;
    return clampf(((x - r.a.x) * dx + (z - r.a.z) * dz) / l2, 0, 1);
}
static float railLength(const Rail& r) { return len(r.b - r.a); }

struct GroundHit { float h, gx, gz; int solid; };
// Highest walkable surface at (x,z) that is not above maxY.
static GroundHit queryGround(float x, float z, float maxY) {
    GroundHit g;
    g.h = baseGround(x, z); g.gx = g.gz = 0; g.solid = -1;
    for (size_t i = 0; i < g_solids.size(); ++i) {
        const Solid& s = g_solids[i];
        if (x < s.x0 || x > s.x1 || z < s.z0 || z > s.z1) continue;
        float t = solidTop(s, x, z);
        if (t <= maxY && t > g.h) {
            g.h = t;
            g.solid = (int)i;
            solidSlope(s, g.gx, g.gz);
        }
    }
    return g;
}

// Pushes a circle (x,z,radius) at height y out of every solid whose surface near it is higher than
// stepTop.  Returns true on contact and the summed outward normal.
static bool resolveWalls(float& x, float& z, float y, float stepTop, float radius, float& nx, float& nz) {
    bool hit = false;
    nx = nz = 0;
    for (int iter = 0; iter < 2; ++iter)
        for (const Solid& s : g_solids) {
            if (x <= s.x0 - radius || x >= s.x1 + radius || z <= s.z0 - radius || z >= s.z1 + radius) continue;
            if (y + 1.6f < s.y0) continue;  // passes underneath (scaffolding, awnings)
            float cx = clampf(x, s.x0, s.x1), cz = clampf(z, s.z0, s.z1);
            if (solidTop(s, cx, cz) <= stepTop) continue;
            float pl = x - (s.x0 - radius), pr = (s.x1 + radius) - x;
            float pb = z - (s.z0 - radius), pf = (s.z1 + radius) - z;
            float m = std::min(std::min(pl, pr), std::min(pb, pf));
            if (m == pl) { x = s.x0 - radius; nx -= 1; }
            else if (m == pr) { x = s.x1 + radius; nx += 1; }
            else if (m == pb) { z = s.z0 - radius; nz -= 1; }
            else { z = s.z1 + radius; nz += 1; }
            hit = true;
        }
    float l = len2D(nx, nz);
    if (l > 0) { nx /= l; nz /= l; }
    return hit;
}
// True if a point is inside a camera-blocking solid (buildings, platform).
static bool cameraBlocked(const V3& p) {
    for (const Solid& s : g_solids) {
        if (!s.camBlock) continue;
        const float m = 0.35f;
        if (p.x > s.x0 - m && p.x < s.x1 + m && p.z > s.z0 - m && p.z < s.z1 + m && p.y < s.yHigh + m && p.y > s.y0 - m)
            return true;
    }
    return false;
}

// ---- Static props (rendered from these lists; collision solids added alongside) ----
enum PropType {
    P_LAMP, P_TREE, P_HYDRANT, P_NEWSBOX, P_MAILBOX, P_PAYPHONE, P_TRASH, P_BARRICADE,
    P_PARKED_CAR, P_PROM_LAMP, P_BENCH_WOOD, P_HOTDOG, P_MANHOLE, P_DUMPSTER, P_PLANTER_TREE
};
struct Prop { int type; float x, z; float rot; int variant; };
static std::vector<Prop> g_props;
struct Puddle { float x, z, rx, rz; };
static std::vector<Puddle> g_puddles;
static void addProp(int type, float x, float z, float rot = 0, int variant = 0) {
    Prop p; p.type = type; p.x = x; p.z = z; p.rot = rot; p.variant = variant;
    g_props.push_back(p);
}

// Notable locations shared by gameplay and rendering.
static const V3 FOUNTAIN_C(0, 0, 26);
static const V3 HYDRANT_POS(6.0f, CURB_H, -8.6f);
static const float PLATFORM_TOP = 1.35f;

// ----------------------------------------------------------------------------
//  World construction
// ----------------------------------------------------------------------------
static const Col C_GRANITE(0.74f, 0.72f, 0.70f);
static const Col C_CONCRETE(0.70f, 0.69f, 0.66f);
static const Col C_WOOD(0.86f, 0.70f, 0.48f);
static const Col C_METAL(0.62f, 0.64f, 0.68f);
static const Col C_BANKBRICK(0.66f, 0.34f, 0.27f);

struct Building {
    float x0, x1, h;
    Col wall;
    const char* sign;
    Col signBg, signFg, awning;  // awning.r < 0: none
    bool fireEscape, waterTower;
};
static const Col NOAWN(-1, 0, 0);
static const Building NORTH_ROW[] = {
    {-82, -62, 21, Col(0.45f, 0.30f, 0.26f), "", Col(), Col(), NOAWN, true, true},
    {-62, -50, 18, Col(0.62f, 0.33f, 0.26f), "ORIGINAL RAY'S PIZZA", Col(0.75f, 0.10f, 0.08f), Col(1.0f, 0.95f, 0.80f), Col(0.10f, 0.45f, 0.20f), true, false},
    {-50, -38, 26, Col(0.70f, 0.58f, 0.44f), "DELI & GROCERY 24 HRS", Col(0.95f, 0.85f, 0.20f), Col(0.75f, 0.10f, 0.10f), Col(0.12f, 0.42f, 0.20f), true, true},
    {-38, -24, 15, Col(0.50f, 0.36f, 0.30f), "VIDEO RENTAL - DVD & VHS", Col(0.10f, 0.20f, 0.60f), Col(1.0f, 0.85f, 0.10f), NOAWN, false, false},
    {-24, -12, 22, Col(0.55f, 0.55f, 0.56f), "99 CENT DREAMS", Col(0.95f, 0.80f, 0.10f), Col(0.80f, 0.10f, 0.10f), Col(0.80f, 0.10f, 0.10f), true, false},
    {-12, 0, 32, Col(0.55f, 0.22f, 0.18f), "LAUNDROMAT", Col(0.92f, 0.92f, 0.90f), Col(0.10f, 0.30f, 0.75f), NOAWN, true, true},
    {0, 12, 19, Col(0.72f, 0.62f, 0.50f), "CELL PHONES & BEEPERS", Col(0.80f, 0.08f, 0.10f), Col(1.0f, 0.90f, 0.20f), NOAWN, false, false},
    {12, 25, 24, Col(0.42f, 0.28f, 0.22f), "VINYL RECORDS", Col(0.08f, 0.08f, 0.08f), Col(1.0f, 0.35f, 0.70f), Col(0.12f, 0.12f, 0.12f), true, true},
    {25, 40, 30, Col(0.60f, 0.58f, 0.55f), "BAGELS & COFFEE", Col(0.90f, 0.50f, 0.10f), Col(0.25f, 0.12f, 0.05f), NOAWN, false, true},
    {40, 52, 17, Col(0.66f, 0.30f, 0.25f), "NAILS & SPA", Col(1.0f, 0.60f, 0.80f), Col(1.0f, 1.0f, 1.0f), Col(0.90f, 0.40f, 0.60f), true, false},
    {52, 64, 22, Col(0.48f, 0.40f, 0.34f), "CHECK CASHING", Col(0.10f, 0.45f, 0.20f), Col(1.0f, 1.0f, 1.0f), NOAWN, false, true},
    {64, 82, 20, Col(0.58f, 0.48f, 0.40f), "", Col(), Col(), NOAWN, true, false},
};
static const int NORTH_ROW_N = sizeof(NORTH_ROW) / sizeof(NORTH_ROW[0]);
static const float SOUTH_BLDG_H = 24.0f;
static const float SCAFFOLD_X0 = 25.0f, SCAFFOLD_X1 = 40.0f;
static const float PARKED_X[] = {-46.0f, -39.5f, -33.0f, -18.0f, -11.5f, 14.0f, 20.5f, 34.0f, 44.0f};
static const int PARKED_N = sizeof(PARKED_X) / sizeof(PARKED_X[0]);
static const float PARK_Z = -6.8f;
static std::vector<int> g_carRoofSolids;  // solid indices that count as parked-car roofs

// Posts every ~3 m along a pipe rail; shared by collision and rendering.
static int railPostCount(const Rail& r) {
    float dx = r.b.x - r.a.x, dz = r.b.z - r.a.z;
    return std::max(1, (int)(len2D(dx, dz) / 3.0f));
}
static void addTubeRail(const V3& a, const V3& b, const char* name) {
    addRail(a, b, RK_METAL, name, true);
    const Rail& r = g_rails.back();
    int n = railPostCount(r);
    for (int k = 0; k <= n; ++k) {
        V3 p = lerpv(a, b, (float)k / n);
        addBox(p.x - 0.06f, p.z - 0.06f, p.x + 0.06f, p.z + 0.06f, 0.0f, p.y - 0.04f, ST_METAL, C_METAL, false);
    }
}
static void addLedgeBox(float x0, float z0, float x1, float z1, float top, int style, Col c, const char* name,
                        bool railX, bool railCentre = false) {
    addBox(x0, z0, x1, z1, 0.0f, top, style, c);
    if (railCentre) {
        float zc = (z0 + z1) * 0.5f;
        addRail(V3(x0 + 0.05f, top, zc), V3(x1 - 0.05f, top, zc), RK_LEDGE, name, false);
    } else if (railX) {  // rails along both long (X) edges
        addRail(V3(x0 + 0.05f, top, z0), V3(x1 - 0.05f, top, z0), RK_LEDGE, name, false);
        addRail(V3(x0 + 0.05f, top, z1), V3(x1 - 0.05f, top, z1), RK_LEDGE, name, false);
    }
}
static void addPostSolid(float x, float z, float r, float h) {
    addBox(x - r, z - r, x + r, z + r, 0.0f, h, ST_METAL, C_METAL, false);
}

static void buildWorld() {
    g_solids.clear(); g_rails.clear(); g_props.clear(); g_puddles.clear(); g_carRoofSolids.clear();

    // --- Boundaries and buildings (rendered separately) ---------------------
    addBox(-WORLD_X - 3, -16, -WORLD_X, 48, -10, 8, ST_CONCRETE, C_CONCRETE, false);
    addBox(WORLD_X, -16, WORLD_X + 3, 48, -10, 8, ST_CONCRETE, C_CONCRETE, false);
    addBox(-90, -45, 90, FACADE_Z, -1, 60, ST_BRICK, C_CONCRETE, false, true);
    addBox(-90, PLAZA_Z0, -WORLD_X, PLAZA_Z1 - 2, -1, SOUTH_BLDG_H, ST_BRICK, C_CONCRETE, false, true);
    addBox(WORLD_X, PLAZA_Z0, 90, PLAZA_Z1 - 2, -1, SOUTH_BLDG_H, ST_BRICK, C_CONCRETE, false, true);

    // --- North sidewalk ------------------------------------------------------
    for (float x : {-45.0f, -15.0f, 15.0f, 45.0f}) { addProp(P_LAMP, x, -8.6f, 0); addPostSolid(x, -8.6f, 0.14f, 6.5f); }
    for (float x : {-36.0f, -6.5f, 50.0f}) { addProp(P_TREE, x, -9.15f); addPostSolid(x, -9.15f, 0.18f, 4.0f); }
    addProp(P_HYDRANT, HYDRANT_POS.x, HYDRANT_POS.z); addPostSolid(HYDRANT_POS.x, HYDRANT_POS.z, 0.17f, 0.85f);
    addProp(P_NEWSBOX, -21.0f, -8.9f, 0, 0); addProp(P_NEWSBOX, -20.4f, -8.9f, 0, 1); addProp(P_NEWSBOX, -19.8f, -8.9f, 0, 2);
    addBox(-21.3f, -9.2f, -19.5f, -8.6f, 0, 1.15f, ST_METAL, C_METAL, false);
    addProp(P_MAILBOX, -26.0f, -8.9f); addBox(-26.3f, -9.15f, -25.7f, -8.65f, 0, 1.25f, ST_METAL, C_METAL, false);
    addProp(P_PAYPHONE, -1.0f, -8.9f); addBox(-1.35f, -9.1f, -0.65f, -8.7f, 0, 2.2f, ST_METAL, C_METAL, false);
    addProp(P_TRASH, 10.0f, -8.9f); addPostSolid(10.0f, -8.9f, 0.3f, 0.95f);
    // Granite bank ledge in front of the laundromat.
    addBox(-10.0f, -13.6f, -2.0f, -13.0f, 0, 0.55f, ST_GRANITE, C_GRANITE);
    addRail(V3(-9.95f, 0.55f, -13.0f), V3(-2.05f, 0.55f, -13.0f), RK_LEDGE, "Bank Ledge", false);
    // Sidewalk scaffolding posts (deck is visual only).
    for (float x = SCAFFOLD_X0; x <= SCAFFOLD_X1 + 0.01f; x += 3.0f) {
        addPostSolid(x, -9.3f, 0.07f, 3.4f);
        addPostSolid(x, -13.7f, 0.07f, 3.4f);
    }
    // Parked cars: lower body + cabin, both landable.
    for (int i = 0; i < PARKED_N; ++i) {
        float cx = PARKED_X[i];
        addProp(P_PARKED_CAR, cx, PARK_Z, 0, i);
        addBox(cx - 2.2f, PARK_Z - 0.9f, cx + 2.2f, PARK_Z + 0.9f, 0, 0.95f, ST_PAINTED, C_METAL, false);
        g_carRoofSolids.push_back(addBox(cx - 1.1f, PARK_Z - 0.82f, cx + 1.0f, PARK_Z + 0.82f, 0, 1.45f, ST_PAINTED, C_METAL, false));
        g_carRoofSolids.push_back((int)g_solids.size() - 2);
    }

    // --- Street --------------------------------------------------------------
    addProp(P_MANHOLE, -20.0f, -2.8f); addProp(P_MANHOLE, 25.0f, 2.6f); addProp(P_MANHOLE, 48.0f, -2.8f);
    g_puddles.push_back({6.6f, -5.0f, 2.3f, 1.3f});   // hydrant runoff
    g_puddles.push_back({-14.0f, 7.0f, 1.7f, 0.7f});
    g_puddles.push_back({33.0f, 6.6f, 1.2f, 0.9f});
    g_puddles.push_back({24.0f, 40.4f, 1.4f, 0.8f});  // promenade
    // South curb is grindable (only when riding along it).
    addRail(V3(-54.5f, CURB_H, SCURB_Z), V3(54.5f, CURB_H, SCURB_Z), RK_CURB, "Curb", false);

    // --- South sidewalk ------------------------------------------------------
    for (float x : {-40.0f, -10.0f, 20.0f, 50.0f}) { addProp(P_LAMP, x, 8.9f, PI); addPostSolid(x, 8.9f, 0.14f, 6.5f); }
    for (float x : {-47.0f, 12.0f, 42.0f}) { addProp(P_TREE, x, 9.3f); addPostSolid(x, 9.3f, 0.18f, 4.0f); }
    addProp(P_HYDRANT, -34.0f, 8.75f); addPostSolid(-34.0f, 8.75f, 0.17f, 0.85f);
    addProp(P_PAYPHONE, 7.6f, 8.95f); addProp(P_PAYPHONE, 8.4f, 8.95f); addBox(7.25f, 8.75f, 8.75f, 9.15f, 0, 2.2f, ST_METAL, C_METAL, false);
    addProp(P_NEWSBOX, -4.0f, 9.0f, PI, 3); addProp(P_NEWSBOX, -3.4f, 9.0f, PI, 0);
    addBox(-4.3f, 8.7f, -3.1f, 9.3f, 0, 1.15f, ST_METAL, C_METAL, false);
    addProp(P_TRASH, 30.0f, 9.0f); addPostSolid(30.0f, 9.0f, 0.3f, 0.95f);
    // Subway entrance: railings on three sides, open to the east.
    addBox(-26.0f, 8.94f, -22.0f, 9.06f, 0, 1.1f, ST_METAL, C_METAL, false);
    addBox(-26.0f, 11.54f, -22.0f, 11.66f, 0, 1.1f, ST_METAL, C_METAL, false);
    addBox(-26.06f, 8.94f, -25.94f, 11.66f, 0, 1.1f, ST_METAL, C_METAL, false);
    addRail(V3(-25.9f, 1.1f, 9.0f), V3(-22.1f, 1.1f, 9.0f), RK_METAL, "Subway Rail", false);
    addRail(V3(-25.9f, 1.1f, 11.6f), V3(-22.1f, 1.1f, 11.6f), RK_METAL, "Subway Rail", false);

    // --- Plaza: "the Banks" platform (west) ----------------------------------
    {
        int p = addBox(-46.0f, 18.0f, -30.0f, 34.0f, 0, PLATFORM_TOP, ST_GRANITE, C_GRANITE);
        g_solids[p].camBlock = true;
        int b = addRamp(-46.0f, 14.0f, -30.0f, 18.0f, CURB_H, PLATFORM_TOP, RAMP_PZ, C_BANKBRICK);
        g_solids[b].style = ST_BRICK;
        g_solids[b].camBlock = true;
        // Six-riser stair set descending east.
        const float tread = 4.0f / 6.0f;
        for (int i = 0; i < 5; ++i)
            addBox(-30.0f + i * tread, 22.0f, -30.0f + (i + 1) * tread, 30.0f, 0, PLATFORM_TOP - (i + 1) * 0.2f,
                   ST_GRANITE, C_GRANITE);
        addTubeRail(V3(-30.6f, 2.38f, 22.3f), V3(-26.6f, 1.18f, 22.3f), "Banks Handrail");
        addTubeRail(V3(-30.6f, 2.38f, 29.7f), V3(-26.6f, 1.18f, 29.7f), "Banks Handrail");
        addLedgeBox(-44.0f, 30.6f, -36.0f, 31.6f, PLATFORM_TOP + 0.45f, ST_GRANITE, C_GRANITE * 0.92f, "Banks Ledge", true);
        addRail(V3(-45.9f, PLATFORM_TOP, 34.0f), V3(-30.1f, PLATFORM_TOP, 34.0f), RK_LEDGE, "Banks Edge", false);
    }

    // --- Plaza: fountain ------------------------------------------------------
    {
        const float cx = FOUNTAIN_C.x, cz = FOUNTAIN_C.z, R = 4.0f, T = 0.4f, top = 0.6f;
        addBox(cx - R, cz - R, cx + R, cz - R + T, 0, top, ST_GRANITE, C_GRANITE);
        addBox(cx - R, cz + R - T, cx + R, cz + R, 0, top, ST_GRANITE, C_GRANITE);
        addBox(cx - R, cz - R + T, cx - R + T, cz + R - T, 0, top, ST_GRANITE, C_GRANITE);
        addBox(cx + R - T, cz - R + T, cx + R, cz + R - T, 0, top, ST_GRANITE, C_GRANITE);
        const float h = T * 0.5f;
        addRail(V3(cx - R, top, cz - R + h), V3(cx + R, top, cz - R + h), RK_LEDGE, "Fountain Ledge", false);
        addRail(V3(cx - R, top, cz + R - h), V3(cx + R, top, cz + R - h), RK_LEDGE, "Fountain Ledge", false);
        addRail(V3(cx - R + h, top, cz - R), V3(cx - R + h, top, cz + R), RK_LEDGE, "Fountain Ledge", false);
        addRail(V3(cx + R - h, top, cz - R), V3(cx + R - h, top, cz + R), RK_LEDGE, "Fountain Ledge", false);
        addBox(cx - 0.9f, cz - 0.9f, cx + 0.9f, cz + 0.9f, 0, 1.35f, ST_GRANITE, C_GRANITE, false);  // pedestal + bowl
    }

    // --- Plaza: benches and planters -----------------------------------------
    for (float x : {-13.5f, 0.0f, 13.5f})
        addLedgeBox(x - 1.5f, 35.2f, x + 1.5f, 35.8f, 0.55f, ST_GRANITE, C_GRANITE * 0.95f, "Bench", false, true);
    for (float x : {-13.0f, 13.0f}) {
        addLedgeBox(x - 2.0f, 25.0f, x + 2.0f, 27.0f, 0.7f, ST_GRANITE, C_GRANITE * 0.9f, "Planter", true);
        addProp(P_PLANTER_TREE, x, 26.0f);
        addPostSolid(x, 26.0f, 0.18f, 4.5f);
    }

    // --- Plaza: east street-skate area ---------------------------------------
    addRamp(26.0f, 16.5f, 28.5f, 18.5f, CURB_H, 0.75f, RAMP_PX, C_WOOD);  // kicker -> manual pad line
    addLedgeBox(33.0f, 16.8f, 41.0f, 18.2f, 0.4f, ST_CONCRETE, C_CONCRETE, "Manual Pad", true);
    addTubeRail(V3(28.0f, 0.55f, 26.0f), V3(40.0f, 0.55f, 26.0f), "Flat Bar");  // 4 m run-out before the kicker's back
    addRamp(44.0f, 25.0f, 46.5f, 27.0f, CURB_H, 0.8f, RAMP_NX, C_WOOD);  // kicker -> flat bar
    addLedgeBox(28.0f, 32.5f, 40.0f, 33.3f, 0.5f, ST_GRANITE, C_GRANITE, "Hubba Ledge", true);
    addProp(P_DUMPSTER, 51.5f, 19.0f);
    addBox(50.0f, 18.0f, 53.0f, 20.0f, 0, 1.3f, ST_PAINTED, Col(0.15f, 0.35f, 0.2f), false);
    addRail(V3(50.05f, 1.3f, 18.0f), V3(52.95f, 1.3f, 18.0f), RK_METAL, "Dumpster", false);

    // --- Waterfront promenade -----------------------------------------------
    addBox(-WORLD_X, 43.55f, WORLD_X, 43.75f, 0, 1.15f, ST_METAL, C_METAL, false);
    addRail(V3(-54.5f, 1.15f, 43.65f), V3(54.5f, 1.15f, 43.65f), RK_METAL, "Hudson Rail", false);
    for (float x : {-40.0f, -20.0f, 0.0f, 20.0f, 40.0f}) { addProp(P_PROM_LAMP, x, 38.6f); addPostSolid(x, 38.6f, 0.12f, 4.5f); }
    for (float x : {-30.0f, -10.0f, 10.0f, 30.0f}) {
        addProp(P_BENCH_WOOD, x, 42.6f);
        addBox(x - 1.0f, 42.3f, x + 1.0f, 42.9f, 0, 0.95f, ST_WOOD, C_WOOD, false);
    }
    addProp(P_HOTDOG, -5.0f, 39.2f);
    addBox(-6.0f, 38.7f, -4.0f, 39.7f, 0, 1.2f, ST_METAL, C_METAL, false);

    // --- End-of-block police barricades (visual; collision is the boundary) --
    for (float s : {-1.0f, 1.0f}) {
        float x = s * (WORLD_X - 0.4f);
        for (float z : {-12.6f, -10.2f, -6.8f, 6.6f, 10.4f, 12.8f, 39.4f, 42.2f}) addProp(P_BARRICADE, x, z, PI * 0.5f);
    }
}

// ----------------------------------------------------------------------------
//  Static scene: ground, buildings, skyline (compiled into display lists)
// ----------------------------------------------------------------------------
static const V3 SUN_DIR = normalize(V3(-0.5f, 0.62f, 0.6f));  // toward the sun: low, south-west
static const float EXT = 420.0f;                              // visual extent of the block east/west (fades into fog)
static inline void texOn(GLuint t) { glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, t); }
static inline void texOff() { glDisable(GL_TEXTURE_2D); }
static inline void unlitOn() { glDisable(GL_LIGHTING); }
static inline void unlitOff() { glEnable(GL_LIGHTING); }

static void drawSolid(const Solid& s) {
    float tile = 1.0f;
    switch (s.style) {
        case ST_GRANITE: texOn(g_texGranite); tile = 1.0f; break;
        case ST_CONCRETE: texOn(g_texConcrete); tile = 1.5f; break;
        case ST_WOOD: texOn(g_texWood); tile = 1.2f; break;
        case ST_BRICK: texOn(g_texBrick); tile = 2.0f; break;
        default: texOff(); break;
    }
    glc(s.col);
    if (s.ramp == RAMP_NONE) {
        drawBox(s.x0, s.y0, s.z0, s.x1, s.yHigh, s.z1, tile, s.style == ST_BRICK ? 1.2f : tile);
        return;
    }
    const float xs[4] = {s.x0, s.x0, s.x1, s.x1}, zs[4] = {s.z0, s.z1, s.z1, s.z0};
    float ys[4];
    V3 p[4];
    for (int i = 0; i < 4; ++i) { ys[i] = solidTop(s, xs[i], zs[i]); p[i] = V3(xs[i], ys[i], zs[i]); }
    glBegin(GL_QUADS);
    nrm(normalize(cross(p[1] - p[0], p[3] - p[0])));
    for (int i = 0; i < 4; ++i) { glTexCoord2f(xs[i] / tile, zs[i] / tile); vtx(p[i]); }
    const V3 ns[4] = {V3(-1, 0, 0), V3(0, 0, 1), V3(1, 0, 0), V3(0, 0, -1)};
    for (int e = 0; e < 4; ++e) {
        int a = e, b = (e + 1) % 4;
        nrm(ns[e]);
        float ua = (e % 2 == 0) ? zs[a] : xs[a], ub = (e % 2 == 0) ? zs[b] : xs[b];
        glTexCoord2f(ua / tile, s.y0 / tile); glVertex3f(xs[a], s.y0, zs[a]);
        glTexCoord2f(ub / tile, s.y0 / tile); glVertex3f(xs[b], s.y0, zs[b]);
        glTexCoord2f(ub / tile, ys[b] / tile); glVertex3f(xs[b], ys[b], zs[b]);
        glTexCoord2f(ua / tile, ys[a] / tile); glVertex3f(xs[a], ys[a], zs[a]);
    }
    glEnd();
    texOff();  // metal lip on the kicker
    glc(Col(0.55f, 0.57f, 0.6f));
    if (s.ramp == RAMP_PX) drawBox(s.x1 - 0.12f, s.yHigh - 0.02f, s.z0, s.x1, s.yHigh + 0.005f, s.z1);
    if (s.ramp == RAMP_NX) drawBox(s.x0, s.yHigh - 0.02f, s.z0, s.x0 + 0.12f, s.yHigh + 0.005f, s.z1);
}

// Vertical strip facing +/-Z, tessellated along X.
static void wallStripZ(float z, float nz, float x0, float x1, float y0, float y1) {
    glNormal3f(0, 0, nz);
    glBegin(GL_QUADS);
    for (float x = x0; x < x1; x += 8.0f) {
        float xe = std::min(x + 8.0f, x1);
        glTexCoord2f(x, y0); glVertex3f(x, y0, z);
        glTexCoord2f(xe, y0); glVertex3f(xe, y0, z);
        glTexCoord2f(xe, y1); glVertex3f(xe, y1, z);
        glTexCoord2f(x, y1); glVertex3f(x, y1, z);
    }
    glEnd();
}

static void drawGroundAll() {
    texOn(g_texAsphalt); glc(Col(0.40f, 0.40f, 0.42f));
    drawGround(-EXT, NCURB_Z, EXT, SCURB_Z, 0.0f, 6.0f);
    texOn(g_texConcrete); glc(Col(0.74f, 0.72f, 0.69f));
    drawGround(-EXT, FACADE_Z, EXT, NCURB_Z, CURB_H, 1.5f);
    drawGround(-EXT, SCURB_Z, -25.94f, PLAZA_Z0, CURB_H, 1.5f);  // south sidewalk, around the subway stairwell
    drawGround(-22.0f, SCURB_Z, EXT, PLAZA_Z0, CURB_H, 1.5f);
    drawGround(-25.94f, SCURB_Z, -22.0f, 9.06f, CURB_H, 1.5f);
    drawGround(-25.94f, 11.54f, -22.0f, PLAZA_Z0, CURB_H, 1.5f);
    texOn(g_texGranite); glc(Col(0.6f, 0.6f, 0.6f));
    wallStripZ(NCURB_Z, 1, -EXT, EXT, 0, CURB_H);
    wallStripZ(SCURB_Z, -1, -EXT, EXT, 0, CURB_H);
    glc(Col(0.78f, 0.76f, 0.73f));
    drawGround(-WORLD_X, PLAZA_Z0, WORLD_X, PLAZA_Z1, CURB_H, 1.0f);
    texOn(g_texConcrete); glc(Col(0.7f, 0.68f, 0.64f));
    drawGround(-EXT, 36.0f, -WORLD_X, PLAZA_Z1, CURB_H, 1.5f);
    drawGround(WORLD_X, 36.0f, EXT, PLAZA_Z1, CURB_H, 1.5f);
    glc(Col(0.74f, 0.62f, 0.52f));  // promenade pavers
    drawGround(-EXT, PLAZA_Z1, EXT, BULKHEAD_Z, CURB_H, 1.0f);
    texOn(g_texGranite); glc(Col(0.5f, 0.48f, 0.45f));
    wallStripZ(BULKHEAD_Z, 1, -EXT, EXT, WATER_Y - 2.5f, CURB_H);
    texOff();
    glc(Col(0.66f, 0.64f, 0.6f));
    drawBox(-EXT, CURB_H, 43.85f, EXT, CURB_H + 0.12f, BULKHEAD_Z);
    glc(Col(0.3f, 0.3f, 0.31f));  // underlay beyond the north buildings
    drawGround(-700, -500, 700, -40, 0.0f, 10.0f, 60.0f);
}

static void drawMarkings() {
    texOff();
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1, -2);
    glc(Col(0.9f, 0.72f, 0.12f));
    drawGround(-EXT, -0.2f, EXT, -0.08f, 0.0f, 1, 10);
    drawGround(-EXT, 0.08f, EXT, 0.2f, 0.0f, 1, 10);
    glc(Col(0.88f, 0.88f, 0.85f));
    drawGround(-EXT, -5.62f, EXT, -5.5f, 0.0f, 1, 10);
    drawGround(-EXT, 5.2f, EXT, 5.34f, 0.0f, 1, 10);
    for (float cx : {-58.5f, 58.5f})
        for (float z = -7.5f; z < 7.6f; z += 1.25f) drawGround(cx - 2.0f, z, cx + 2.0f, z + 0.6f, 0.0f, 1, 10);
    for (float x = -50.0f; x <= 50.0f; x += 25.0f) {  // bike-lane diamonds
        glNormal3f(0, 1, 0);
        glBegin(GL_LINE_LOOP);
        glVertex3f(x - 1.0f, 0.01f, 6.6f); glVertex3f(x, 0.01f, 6.1f);
        glVertex3f(x + 1.0f, 0.01f, 6.6f); glVertex3f(x, 0.01f, 7.1f);
        glEnd();
    }
    glc(Col(0.25f, 0.5f, 0.55f));  // fountain basin tiles
    drawGround(-3.6f, 22.4f, 3.6f, 29.6f, CURB_H, 1, 4);
    glDisable(GL_POLYGON_OFFSET_FILL);
}

// ---- Facades ------------------------------------------------------------------
struct Facade {
    float w, h;
    const char* sign;
    Col signBg, signFg, awning;
    bool store, fireEscape;
    const char* mural;
    int seed;
};
static void gradQuad(float x0, float y0, float x1, float y1, float z, const Col& top, const Col& bot) {
    glNormal3f(0, 0, 1);
    glBegin(GL_QUADS);
    glc(bot); glVertex3f(x0, y0, z); glVertex3f(x1, y0, z);
    glc(top); glVertex3f(x1, y1, z); glVertex3f(x0, y1, z);
    glEnd();
}
static const Col PAINT[6] = {Col(1.0f, 0.85f, 0.1f), Col(1.0f, 0.3f, 0.55f), Col(0.3f, 1.0f, 0.45f),
                             Col(1.0f, 0.5f, 0.1f), Col(0.4f, 0.8f, 1.0f), Col(0.95f, 0.95f, 0.95f)};

static void drawMural(float x0, float x1, float y0, float y1, const char* text, int seed) {
    unlitOn();
    gradQuad(x0, y0, x1, y1, 0.03f, Col(0.12f, 0.55f, 0.75f), Col(0.25f, 0.1f, 0.38f));
    for (int i = 0; i < 18; ++i) {  // spray-paint splats
        float cx = lerpf(x0 + 0.4f, x1 - 0.4f, hash01(seed, i, 1)), cy = lerpf(y0 + 0.3f, y1 - 0.3f, hash01(seed, i, 2));
        float r = 0.2f + 0.55f * hash01(seed, i, 3);
        const Col& c = PAINT[i % 6];
        glBegin(GL_TRIANGLE_FAN);
        glColor3f(c.r, c.g, c.b); glVertex3f(cx, cy, 0.035f);
        glColor3f(c.r * 0.6f, c.g * 0.6f, c.b * 0.6f);
        for (int k = 0; k <= 14; ++k) {
            float a = 2 * PI * k / 14, rr = r * (0.75f + 0.35f * hash01(seed, i, k + 10));
            glVertex3f(cx + cosf(a) * rr, cy + sinf(a) * rr, 0.035f);
        }
        glEnd();
    }
    float capH = (y1 - y0) * 0.55f, sc = capH / 100.0f, tw = strokeWidth(text) * sc;
    if (tw > (x1 - x0) * 0.9f) { sc *= (x1 - x0) * 0.9f / tw; tw = (x1 - x0) * 0.9f; capH = sc * 100.0f; }
    glLineWidth(3.0f);
    float pen = (x0 + x1) * 0.5f - tw * 0.5f;
    for (int ci = 0; text[ci]; ++ci) {
        int ch = text[ci];
        float wob = 0.12f * sinf(ci * 1.7f + seed);
        for (int pass = 0; pass < 2; ++pass) {
            float r = pass == 0 ? 0.07f : 0.025f;
            if (pass == 0) glColor3f(0.05f, 0.05f, 0.08f); else glc(PAINT[(ci + seed) % 5]);
            for (int ox = -1; ox <= 1; ++ox)
                for (int oy = -1; oy <= 1; ++oy) {
                    glPushMatrix();
                    glTranslatef(pen + ox * r, (y0 + y1) * 0.5f - capH * 0.5f + wob + oy * r, 0.04f + pass * 0.006f);
                    glScalef(sc, sc, sc);
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, ch);
                    glPopMatrix();
                }
        }
        pen += glutStrokeWidth(GLUT_STROKE_ROMAN, ch) * sc;
    }
    glLineWidth(1.0f);
    unlitOff();
}

// Facade detail in a local frame: wall plane z = 0, outward +Z, x in [0,w], y up.
static void drawFacade(const Facade& f) {
    texOff();
    const Col trim(0.80f, 0.77f, 0.70f), iron(0.10f, 0.10f, 0.10f);
    float upperStart = 5.0f;
    if (f.store) {
        upperStart = 4.8f;
        glc(Col(0.55f, 0.52f, 0.48f));
        drawBox(0.0f, 0.0f, 0.0f, 0.35f, 4.6f, 0.2f);
        drawBox(f.w - 0.35f, 0.0f, 0.0f, f.w, 4.6f, 0.2f);
        glc(Col(0.22f, 0.2f, 0.19f));
        drawBox(0.35f, 0.0f, 0.0f, f.w - 0.35f, 0.6f, 0.12f);
        const float d0 = f.w - 2.4f, d1 = f.w - 1.2f;
        unlitOn();
        const Col gTop(0.52f, 0.58f, 0.64f), gBot(0.78f, 0.64f, 0.44f);
        gradQuad(0.35f, 0.6f, d0 - 0.1f, 3.25f, 0.05f, gTop, gBot);
        gradQuad(d1 + 0.1f, 0.6f, f.w - 0.35f, 3.25f, 0.05f, gTop, gBot);
        gradQuad(d0, 0.0f, d1, 2.5f, 0.05f, Col(0.3f, 0.33f, 0.36f), Col(0.45f, 0.38f, 0.3f));
        if (hash01(f.seed, 9) < 0.6f) {  // neon OPEN sign
            glColor3f(1.0f, 0.15f, 0.2f);
            glLineWidth(2.0f);
            glPushMatrix();
            glTranslatef(1.3f, 2.4f, 0.07f);
            drawStrokeCentered("OPEN", 0.22f, 0, 0.0f);
            glPopMatrix();
            glLineWidth(1.0f);
        }
        unlitOff();
        glc(Col(0.15f, 0.15f, 0.16f));
        for (float x = 0.35f; x < d0 - 0.2f; x += 1.7f) drawBox(x, 0.6f, 0.0f, x + 0.08f, 3.3f, 0.09f);
        drawBox(0.35f, 3.25f, 0.0f, f.w - 0.35f, 3.55f, 0.1f);
        drawBox(d0 - 0.1f, 0.0f, 0.0f, d0, 3.25f, 0.1f);
        drawBox(d1, 0.0f, 0.0f, d1 + 0.1f, 3.25f, 0.1f);
        drawBox(d0, 2.5f, 0.0f, d1, 2.6f, 0.1f);
        glc(f.signBg);
        drawBox(0.35f, 3.55f, 0.0f, f.w - 0.35f, 4.5f, 0.25f);
        if (f.sign && f.sign[0]) {
            unlitOn();
            glc(f.signFg);
            glLineWidth(2.0f);
            glPushMatrix();
            glTranslatef(f.w * 0.5f, 4.02f, 0.27f);
            drawStrokeCentered(f.sign, 0.52f, f.w - 1.4f, 0.012f);
            glPopMatrix();
            glLineWidth(1.0f);
            unlitOff();
        }
        if (f.awning.r >= 0) {
            const float yT = 3.5f, yB = 2.8f, zo = 1.5f;
            V3 n = normalize(V3(0, zo, yT - yB));
            glBegin(GL_QUADS);
            int k = 0;
            for (float x = 0.45f; x < f.w - 0.46f; x += 0.6f, ++k) {
                float xe = std::min(x + 0.6f, f.w - 0.45f);
                glc((k & 1) ? Col(0.92f, 0.9f, 0.85f) : f.awning);
                nrm(n);
                glVertex3f(x, yT, 0.02f); glVertex3f(x, yB, zo); glVertex3f(xe, yB, zo); glVertex3f(xe, yT, 0.02f);
                glNormal3f(0, 0, 1);
                glVertex3f(x, yB, zo); glVertex3f(x, yB - 0.3f, zo); glVertex3f(xe, yB - 0.3f, zo); glVertex3f(xe, yB, zo);
            }
            glEnd();
        }
    }
    const float floorH = 3.2f;
    int floors = std::max(0, (int)((f.h - upperStart - 1.2f) / floorH));
    int cols = std::max(1, (int)((f.w - 0.8f) / 2.5f));
    float pitch = f.w / cols;
    for (int fl = 0; fl < floors; ++fl) {
        float wy0 = upperStart + fl * floorH + 0.7f, wy1 = wy0 + 1.75f;
        for (int c = 0; c < cols; ++c) {
            float cx = pitch * (c + 0.5f), hw = 0.55f, hv = hash01(f.seed, fl, c);
            glc(trim);
            drawBox(cx - hw - 0.15f, wy1, 0.0f, cx + hw + 0.15f, wy1 + 0.25f, 0.12f);
            drawBox(cx - hw - 0.12f, wy0 - 0.12f, 0.0f, cx + hw + 0.12f, wy0, 0.18f);
            unlitOn();
            if (hv < 0.14f) gradQuad(cx - hw, wy0, cx + hw, wy1, 0.03f, Col(1.0f, 0.86f, 0.6f), Col(0.9f, 0.7f, 0.45f));
            else if (hv < 0.3f) gradQuad(cx - hw, wy0, cx + hw, wy1, 0.03f, Col(0.78f, 0.7f, 0.6f), Col(0.62f, 0.52f, 0.44f));
            else gradQuad(cx - hw, wy0, cx + hw, wy1, 0.03f, Col(0.58f, 0.66f, 0.76f), Col(0.2f, 0.23f, 0.28f));
            unlitOff();
            glc(Col(0.88f, 0.86f, 0.82f));
            drawBox(cx - hw, (wy0 + wy1) * 0.5f - 0.04f, 0.0f, cx + hw, (wy0 + wy1) * 0.5f + 0.04f, 0.06f);
            if (hash01(f.seed, fl, c + 50) < 0.12f) {  // window AC unit
                glc(Col(0.8f, 0.8f, 0.77f));
                drawBox(cx - 0.36f, wy0, 0.0f, cx + 0.36f, wy0 + 0.42f, 0.5f);
            }
        }
    }
    glc(Col(0.48f, 0.45f, 0.42f));
    drawBox(0.02f, f.h - 0.7f, 0.0f, f.w - 0.02f, f.h + 0.15f, 0.55f);
    glc(Col(0.62f, 0.6f, 0.56f));
    drawBox(0.02f, f.h - 1.05f, 0.0f, f.w - 0.02f, f.h - 0.7f, 0.28f);
    if (f.fireEscape && floors > 1) {
        const float fx0 = f.w * 0.5f - 2.3f, fx1 = f.w * 0.5f + 2.3f, d = 1.1f;
        glc(iron);
        for (int fl = 0; fl < floors; ++fl) {
            float py = upperStart + fl * floorH + 0.55f;
            drawBox(fx0, py - 0.06f, 0.0f, fx1, py, d);
            drawBox(fx0, py + 0.92f, d - 0.05f, fx1, py + 0.98f, d);
            drawBox(fx0, py + 0.92f, 0.0f, fx0 + 0.05f, py + 0.98f, d);
            drawBox(fx1 - 0.05f, py + 0.92f, 0.0f, fx1, py + 0.98f, d);
            for (float bx = fx0; bx <= fx1 + 0.001f; bx += 0.35f) drawBox(bx, py, d - 0.04f, bx + 0.03f, py + 0.92f, d - 0.01f);
            if (fl + 1 < floors) {
                V3 a(fx0 + 0.5f, py, 0.3f), b(fx1 - 1.3f, py + floorH, 0.3f);
                drawCyl(a, b, 0.025f, 0.025f, 4, false);
                drawCyl(a + V3(0, 0, 0.5f), b + V3(0, 0, 0.5f), 0.025f, 0.025f, 4, false);
                for (int s = 1; s < 10; ++s) {
                    V3 p = lerpv(a, b, s / 10.0f);
                    drawBox(p.x - 0.12f, p.y - 0.02f, 0.3f, p.x + 0.12f, p.y, 0.8f);
                }
            }
        }
    }
    if (f.mural) drawMural(1.5f, f.w - 1.5f, 0.5f, 4.6f, f.mural, f.seed);
}
static void placeFacade(float ox, float oz, float rotDeg, const Facade& f) {
    glPushMatrix();
    glTranslatef(ox, 0, oz);
    glRotatef(rotDeg, 0, 1, 0);
    drawFacade(f);
    glPopMatrix();
}
static void drawBrickBody(float x0, float z0, float x1, float z1, float h, const Col& c) {
    texOn(g_texBrick); glc(c);
    drawBox(x0, 0, z0, x1, h, z1, 2.0f, 1.2f);
    texOff();
    glc(Col(0.22f, 0.21f, 0.2f));
    drawBox(x0 + 0.05f, h, z0 + 0.05f, x1 - 0.05f, h + 0.06f, z1 - 0.05f);
}
static void drawWaterTower(float cx, float cz, float base) {
    texOff();
    glc(Col(0.15f, 0.15f, 0.15f));
    for (int i = 0; i < 4; ++i) {
        float x = cx + ((i & 1) ? 1.0f : -1.0f), z = cz + ((i & 2) ? 1.0f : -1.0f);
        drawCyl(V3(x, base, z), V3(x, base + 3.0f, z), 0.07f, 0.07f, 6, false);
    }
    drawBox(cx - 1.4f, base + 3.0f, cz - 1.4f, cx + 1.4f, base + 3.15f, cz + 1.4f);
    glc(Col(0.48f, 0.35f, 0.24f));
    drawCyl(V3(cx, base + 3.15f, cz), V3(cx, base + 6.3f, cz), 1.4f, 1.4f, 16);
    glc(Col(0.12f, 0.12f, 0.12f));
    for (int b = 0; b < 3; ++b) drawCyl(V3(cx, base + 3.7f + b, cz), V3(cx, base + 3.78f + b, cz), 1.43f, 1.43f, 16, false);
    glc(Col(0.32f, 0.26f, 0.2f));
    drawCyl(V3(cx, base + 6.3f, cz), V3(cx, base + 7.5f, cz), 1.55f, 0.06f, 16);
}

static void drawBuildings() {
    for (int i = 0; i < NORTH_ROW_N; ++i) {
        const Building& b = NORTH_ROW[i];
        bool scaffolded = b.x0 == SCAFFOLD_X0;  // its sign lives on the sidewalk shed
        drawBrickBody(b.x0, -40.0f, b.x1, FACADE_Z, b.h, b.wall);
        Facade f{b.x1 - b.x0, b.h, scaffolded ? "" : b.sign, b.signBg, b.signFg, b.awning, true, b.fireEscape, nullptr, i * 7 + 1};
        placeFacade(b.x0, FACADE_Z, 0, f);
        if (b.waterTower) drawWaterTower((b.x0 + b.x1) * 0.5f + 1.5f, -24.0f, b.h);
    }
    for (int side = -1; side <= 1; side += 2)  // filler buildings toward the fog; detailed facades only nearby
        for (int k = 0; k < 25; ++k) {
            float x0 = side > 0 ? 82.0f + k * 14.0f : -96.0f - k * 14.0f;
            float h = 16.0f + 14.0f * hash01(k, side + 5, 3);
            drawBrickBody(x0, -40.0f, x0 + 14.0f, FACADE_Z, h, Col(0.5f, 0.36f, 0.3f) * (0.8f + 0.4f * hash01(k, side + 3, 4)));
            if (k < 5)
                placeFacade(x0, FACADE_Z, 0, Facade{14.0f, h, "", Col(0.15f, 0.25f, 0.18f), Col(), NOAWN, true, hash01(k, side + 3, 5) < 0.5f, nullptr, 100 + k * 3 + side});
        }
    // South flank buildings beside the plaza.
    drawBrickBody(-EXT, PLAZA_Z0, -WORLD_X, 36.0f, SOUTH_BLDG_H, Col(0.52f, 0.32f, 0.27f));
    drawBrickBody(WORLD_X, PLAZA_Z0, EXT, 36.0f, SOUTH_BLDG_H + 3.0f, Col(0.62f, 0.56f, 0.5f));
    const float widths[6] = {14, 21, 15, 15, 15, 15};
    const char* wSigns[6] = {"BODEGA - LOTTO - ATM", "LIQUORS", "", "", "", ""};
    const char* eSigns[6] = {"CHINESE TAKE OUT", "HARDWARE", "", "", "", ""};
    const Col wBg[2] = {Col(0.95f, 0.95f, 0.9f), Col(0.1f, 0.1f, 0.1f)}, wFg[2] = {Col(0.1f, 0.2f, 0.6f), Col(1.0f, 0.3f, 0.2f)};
    const Col eBg[2] = {Col(0.85f, 0.1f, 0.1f), Col(0.1f, 0.3f, 0.6f)}, eFg[2] = {Col(1.0f, 0.9f, 0.3f), Col(1, 1, 1)};
    const Col plainBg(0.15f, 0.25f, 0.18f);
    float xw = -WORLD_X, xe = WORLD_X;
    for (int i = 0; i < 6; ++i) {
        // Face toward the street (-Z): rotate 180, origin at the chunk's east end.
        placeFacade(xw, PLAZA_Z0, 180, Facade{widths[i], SOUTH_BLDG_H, wSigns[i], i < 2 ? wBg[i] : plainBg, i < 2 ? wFg[i] : Col(),
                                             i == 0 ? Col(0.75f, 0.1f, 0.1f) : NOAWN, true, i % 2 == 1, nullptr, 300 + i});
        placeFacade(xe + widths[i], PLAZA_Z0, 180, Facade{widths[i], SOUTH_BLDG_H + 3.0f, eSigns[i], i < 2 ? eBg[i] : plainBg, i < 2 ? eFg[i] : Col(),
                                                         i == 1 ? Col(0.1f, 0.3f, 0.6f) : NOAWN, true, i % 2 == 0, nullptr, 320 + i});
        // Face toward the promenade (+Z).
        placeFacade(xw - widths[i], 36.0f, 0, Facade{widths[i], SOUTH_BLDG_H, "", plainBg, Col(), NOAWN, true, false, nullptr, 340 + i});
        placeFacade(xe, 36.0f, 0, Facade{widths[i], SOUTH_BLDG_H + 3.0f, "", plainBg, Col(), NOAWN, true, false, nullptr, 360 + i});
        xw -= widths[i];
        xe += widths[i];
    }
    // Plaza-facing walls with murals.
    placeFacade(-WORLD_X, 36.0f, 90, Facade{22.0f, SOUTH_BLDG_H, "", Col(), Col(), NOAWN, false, false, "NYC SKATE", 400});
    placeFacade(WORLD_X, PLAZA_Z0, -90, Facade{22.0f, SOUTH_BLDG_H + 3.0f, "", Col(), Col(), NOAWN, false, true, "SHRED THE CITY", 401});
}

static void drawSkyline() {
    texOff();
    for (int i = 0; i < 70; ++i) {  // midtown to the north
        float x = -330.0f + i * 9.6f + hash01(i, 1, 7) * 6.0f, z = -60.0f - hash01(i, 2, 7) * 200.0f;
        float w = 10.0f + hash01(i, 3, 7) * 14.0f, d = 10.0f + hash01(i, 4, 7) * 14.0f;
        float h = 25.0f + powf(hash01(i, 5, 7), 1.6f) * 120.0f, g = 0.5f + hash01(i, 6, 7) * 0.22f;
        glColor3f(g * 0.92f, g * 0.94f, g);
        drawBox(x, 0, z - d, x + w, h, z);
    }
    const float tx = 70.0f, tz = -230.0f;  // art-deco landmark tower
    glc(Col(0.68f, 0.66f, 0.62f));
    drawBox(tx - 22, 0, tz - 18, tx + 22, 80, tz + 18);
    drawBox(tx - 15, 80, tz - 12, tx + 15, 150, tz + 12);
    drawBox(tx - 10, 150, tz - 8, tx + 10, 200, tz + 8);
    drawBox(tx - 6, 200, tz - 5, tx + 6, 225, tz + 5);
    drawBox(tx - 3.5f, 225, tz - 3.5f, tx + 3.5f, 240, tz + 3.5f);
    drawCyl(V3(tx, 240, tz), V3(tx, 275, tz), 1.6f, 0.25f, 8);
    glc(Col(0.42f, 0.44f, 0.4f));  // far shore
    drawBox(-600, -3, 300, 600, 1.5f, 460);
    for (int i = 0; i < 60; ++i) {
        float x = -380.0f + i * 13.0f + hash01(i, 1, 9) * 8.0f, z = 305.0f + hash01(i, 2, 9) * 90.0f;
        float w = 9.0f + hash01(i, 3, 9) * 16.0f, h = 10.0f + powf(hash01(i, 5, 9), 2.0f) * 55.0f;
        float g = 0.55f + hash01(i, 6, 9) * 0.2f;
        glColor3f(g, g * 0.95f, g * 0.9f);
        drawBox(x, 1.5f, z, x + w, h, z + w);
    }
}

static void drawBridge() {
    const float bx0 = 143.0f, bx1 = 157.0f, deckY = 40.0f;
    texOff();
    glc(Col(0.36f, 0.36f, 0.38f));
    drawBox(bx0, deckY - 2.5f, -80.0f, bx1, deckY, 420.0f);
    const float towers[2] = {110.0f, 220.0f};
    for (float tz : towers) {
        glc(Col(0.62f, 0.55f, 0.47f));
        drawBox(bx0 - 5, WATER_Y - 3, tz - 6, bx1 + 5, 6.0f, tz + 6);
        drawBox(bx0 - 3, 6.0f, tz - 3.5f, bx0 + 1.5f, 95.0f, tz + 3.5f);
        drawBox(bx1 - 1.5f, 6.0f, tz - 3.5f, bx1 + 3, 95.0f, tz + 3.5f);
        drawBox(bx0 + 1.5f, 82.0f, tz - 3.0f, bx1 - 1.5f, 95.0f, tz + 3.0f);
        drawBox(bx0 + 1.5f, deckY + 12.0f, tz - 3.0f, bx1 - 1.5f, deckY + 18.0f, tz + 3.0f);
        drawBox(bx0 - 3.2f, 95.0f, tz - 3.7f, bx1 + 3.2f, 97.0f, tz + 3.7f);
    }
    unlitOn();
    glColor3f(0.25f, 0.26f, 0.3f);
    glLineWidth(2.0f);
    auto cableY = [&](float z) {
        if (z < towers[0]) { float t = (z + 40.0f) / (towers[0] + 40.0f); return lerpf(deckY + 2.0f, 95.0f, t) - 10.0f * t * (1 - t); }
        if (z > towers[1]) { float t = (z - towers[1]) / (380.0f - towers[1]); return lerpf(95.0f, deckY + 2.0f, t) - 10.0f * t * (1 - t); }
        float u = (z - 165.0f) / 55.0f;
        return deckY + 4.0f + (95.0f - deckY - 4.0f) * u * u;
    };
    for (float cx : {bx0, bx1}) {
        glBegin(GL_LINE_STRIP);
        for (float z = -40.0f; z <= 380.0f; z += 5.0f) glVertex3f(cx, cableY(z), z);
        glEnd();
        glBegin(GL_LINES);
        for (float z = -30.0f; z <= 370.0f; z += 7.0f) { glVertex3f(cx, cableY(z), z); glVertex3f(cx, deckY, z); }
        glEnd();
    }
    glLineWidth(1.0f);
    unlitOff();
}

// ---- Props ----------------------------------------------------------------------
static const Col CAR_COLS[8] = {Col(0.96f, 0.76f, 0.1f), Col(0.42f, 0.07f, 0.08f), Col(0.12f, 0.16f, 0.34f), Col(0.68f, 0.7f, 0.72f),
                                Col(0.07f, 0.07f, 0.08f), Col(0.88f, 0.88f, 0.86f), Col(0.2f, 0.34f, 0.24f), Col(0.5f, 0.42f, 0.3f)};
static GLuint g_carLists[8];

// Car facing local +X, ground at y = 0.  Colour 0 is a yellow cab.
static void drawCarModel(int ci) {
    const Col c = CAR_COLS[ci & 7];
    texOff();
    glc(c);
    drawBox(-2.25f, 0.3f, -0.9f, 2.25f, 0.95f, 0.9f);
    drawBox(-1.15f, 0.95f, -0.8f, 1.0f, 1.45f, 0.8f);
    glc(Col(0.1f, 0.12f, 0.15f));
    drawBox(-1.05f, 1.0f, -0.81f, 0.92f, 1.4f, 0.81f);
    drawBox(0.98f, 1.0f, -0.72f, 1.01f, 1.4f, 0.72f);
    drawBox(-1.16f, 1.0f, -0.72f, -1.13f, 1.4f, 0.72f);
    glc(c);
    drawBox(-0.12f, 1.0f, -0.815f, -0.02f, 1.4f, 0.815f);
    glc(Col(0.22f, 0.22f, 0.23f));
    drawBox(2.25f, 0.3f, -0.86f, 2.36f, 0.55f, 0.86f);
    drawBox(-2.36f, 0.3f, -0.86f, -2.25f, 0.55f, 0.86f);
    drawBox(2.25f, 0.6f, -0.45f, 2.27f, 0.85f, 0.45f);
    for (int i = 0; i < 4; ++i) {
        float wx = (i & 1) ? 1.45f : -1.45f, wz = (i & 2) ? 0.82f : -0.82f, o = (i & 2) ? 0.1f : -0.1f;
        glc(Col(0.06f, 0.06f, 0.06f));
        drawCyl(V3(wx, 0.32f, wz - 0.1f), V3(wx, 0.32f, wz + 0.1f), 0.32f, 0.32f, 12);
        glc(Col(0.62f, 0.62f, 0.64f));
        drawCyl(V3(wx, 0.32f, wz + o), V3(wx, 0.32f, wz + o * 1.15f), 0.17f, 0.17f, 10);
    }
    unlitOn();
    glColor3f(1.0f, 0.96f, 0.82f);
    drawBox(2.25f, 0.66f, -0.82f, 2.28f, 0.82f, -0.52f);
    drawBox(2.25f, 0.66f, 0.52f, 2.28f, 0.82f, 0.82f);
    glColor3f(0.75f, 0.05f, 0.05f);
    drawBox(-2.28f, 0.66f, -0.84f, -2.25f, 0.84f, -0.58f);
    drawBox(-2.28f, 0.66f, 0.58f, -2.25f, 0.84f, 0.84f);
    if (ci == 0) {
        glColor3f(1.0f, 0.97f, 0.85f);
        drawBox(-0.35f, 1.45f, -0.2f, 0.25f, 1.66f, 0.2f);
        glColor3f(0.05f, 0.05f, 0.05f);
        for (int s = 0; s < 2; ++s) {
            glPushMatrix();
            glTranslatef(-0.05f, 1.555f, 0.0f);
            glRotatef(s * 180.0f, 0, 1, 0);
            glTranslatef(0, 0, 0.205f);
            drawStrokeCentered("TAXI", 0.11f, 0.5f, 0);
            glPopMatrix();
        }
        drawBox(-2.0f, 0.6f, -0.905f, 2.0f, 0.7f, 0.905f);
        glColor3f(0.95f, 0.95f, 0.95f);
        for (float x = -1.9f; x < 1.9f; x += 0.2f) drawBox(x, 0.65f, -0.91f, x + 0.1f, 0.7f, 0.91f);
    }
    unlitOff();
}

static void drawTree(float x, float z, float base, int seed) {
    texOff();
    glc(Col(0.3f, 0.22f, 0.16f));
    drawCyl(V3(x, base, z), V3(x, base + 2.7f, z), 0.15f, 0.09f, 8);
    for (int b = 0; b < 3; ++b) {
        float a = b * 2.1f + hash01(seed, b);
        drawCyl(V3(x, base + 2.2f, z), V3(x + cosf(a) * 0.9f, base + 3.3f, z + sinf(a) * 0.9f), 0.07f, 0.03f, 5, false);
    }
    for (int i = 0; i < 7; ++i) {
        float a = i * 0.9f + hash01(seed, i, 1) * 2.0f, r = (i == 0) ? 0.0f : 0.85f;
        V3 c(x + cosf(a) * r, base + 3.6f + (hash01(seed, i, 2) - 0.5f) * 0.9f, z + sinf(a) * r);
        float g = 0.75f + 0.4f * hash01(seed, i, 3), s = 0.85f + 0.4f * hash01(seed, i, 4);
        glColor3f(0.22f * g, 0.42f * g, 0.16f * g);
        drawSphere(c, s, s * 0.8f, s);
    }
}
static void drawLamp(float x, float z, float rot, bool streetSign) {
    V3 b(x, CURB_H, z), d = fwdFromYaw(rot), top = b + V3(0, 6.3f, 0);
    V3 mid = top + d * 0.9f + V3(0, 0.35f, 0), end = top + d * 2.0f + V3(0, 0.45f, 0);
    glc(Col(0.3f, 0.33f, 0.32f));
    drawCyl(b, b + V3(0, 0.7f, 0), 0.2f, 0.15f, 10);
    drawCyl(b + V3(0, 0.7f, 0), top, 0.11f, 0.08f, 8);
    drawCyl(top, mid, 0.07f, 0.06f, 6);
    drawCyl(mid, end, 0.06f, 0.05f, 6);
    glPushMatrix();
    glTranslatef(end.x + d.x * 0.3f, end.y, end.z + d.z * 0.3f);
    glRotatef(degf(rot), 0, 1, 0);
    glc(Col(0.55f, 0.56f, 0.54f));
    drawSphere(V3(0, 0, 0), 0.25f, 0.13f, 0.52f);
    unlitOn();
    glColor3f(1.0f, 0.95f, 0.78f);
    drawSphere(V3(0, -0.08f, 0.04f), 0.17f, 0.05f, 0.36f);
    unlitOff();
    glPopMatrix();
    if (!streetSign) return;
    glPushMatrix();
    glTranslatef(x, CURB_H + 3.5f, z);
    glc(Col(0.05f, 0.38f, 0.2f));
    drawBox(0.12f, -0.15f, -0.02f, 1.5f, 0.15f, 0.02f);
    unlitOn();
    glColor3f(1, 1, 1);
    for (int s = 0; s < 2; ++s) {
        glPushMatrix();
        glTranslatef(0.81f, 0, 0);
        glRotatef(s * 180.0f, 0, 1, 0);
        glTranslatef(0, 0, 0.025f);
        drawStrokeCentered("WATER ST", 0.16f, 1.2f, 0.004f);
        glPopMatrix();
    }
    unlitOff();
    glPopMatrix();
}
static void drawHydrant(float x, float z, float facing) {
    V3 b(x, CURB_H, z), f = fwdFromYaw(facing), s(f.z, 0, -f.x);
    const Col red(0.75f, 0.12f, 0.1f), cap(0.75f, 0.75f, 0.72f);
    glc(red);
    drawCyl(b, b + V3(0, 0.08f, 0), 0.2f, 0.2f, 10);
    drawCyl(b + V3(0, 0.08f, 0), b + V3(0, 0.62f, 0), 0.13f, 0.12f, 10);
    drawCyl(b + V3(0, 0.62f, 0), b + V3(0, 0.68f, 0), 0.16f, 0.16f, 10);
    V3 n = b + V3(0, 0.45f, 0);
    drawCyl(n, n + f * 0.2f, 0.07f, 0.07f, 8);
    drawCyl(n - s * 0.2f + V3(0, -0.03f, 0), n + s * 0.2f + V3(0, -0.03f, 0), 0.045f, 0.045f, 8);
    glc(cap);
    drawSphere(b + V3(0, 0.7f, 0), 0.13f, 0.11f, 0.13f);
    drawCyl(b + V3(0, 0.78f, 0), b + V3(0, 0.86f, 0), 0.04f, 0.03f, 6);
}
static const Col NEWS_COLS[4] = {Col(0.15f, 0.3f, 0.7f), Col(0.75f, 0.12f, 0.12f), Col(0.92f, 0.75f, 0.12f), Col(0.15f, 0.5f, 0.25f)};
static void drawNewsbox(float x, float z, float facing, int v) {
    glPushMatrix();
    glTranslatef(x, CURB_H, z);
    glRotatef(degf(facing), 0, 1, 0);
    glc(Col(0.2f, 0.2f, 0.2f));
    for (float lx : {-0.22f, 0.22f}) for (float lz : {-0.18f, 0.18f}) drawBox(lx - 0.02f, 0, lz - 0.02f, lx + 0.02f, 0.4f, lz + 0.02f);
    glc(NEWS_COLS[v & 3]);
    drawBox(-0.26f, 0.4f, -0.22f, 0.26f, 1.1f, 0.22f);
    glc(Col(0.82f, 0.8f, 0.74f));
    drawBox(-0.19f, 0.7f, 0.22f, 0.19f, 1.0f, 0.235f);
    glc(Col(0.15f, 0.15f, 0.15f));
    drawBox(-0.06f, 0.55f, 0.22f, 0.06f, 0.63f, 0.24f);
    glPopMatrix();
}
static void drawMailbox(float x, float z, float facing) {
    glPushMatrix();
    glTranslatef(x, CURB_H, z);
    glRotatef(degf(facing), 0, 1, 0);
    glc(Col(0.08f, 0.1f, 0.2f));
    for (float lx : {-0.22f, 0.22f}) for (float lz : {-0.18f, 0.18f}) drawBox(lx - 0.03f, 0, lz - 0.03f, lx + 0.03f, 0.3f, lz + 0.03f);
    glc(Col(0.1f, 0.2f, 0.48f));
    drawBox(-0.27f, 0.3f, -0.23f, 0.27f, 0.92f, 0.23f);
    drawCyl(V3(-0.27f, 0.92f, 0), V3(0.27f, 0.92f, 0), 0.23f, 0.23f, 14);
    glc(Col(0.75f, 0.75f, 0.75f));
    drawBox(-0.15f, 0.82f, 0.23f, 0.15f, 0.88f, 0.26f);
    unlitOn();
    glColor3f(1, 1, 1);
    glPushMatrix();
    glTranslatef(0, 0.55f, 0.235f);
    drawStrokeCentered("U.S. MAIL", 0.07f, 0.45f, 0);
    glPopMatrix();
    unlitOff();
    glPopMatrix();
}
static void drawPayphone(float x, float z, float facing) {
    glPushMatrix();
    glTranslatef(x, CURB_H, z);
    glRotatef(degf(facing), 0, 1, 0);
    glc(Col(0.55f, 0.57f, 0.6f));
    drawBox(-0.06f, 0, -0.06f, 0.06f, 1.0f, 0.06f);
    drawBox(-0.28f, 1.0f, -0.17f, 0.28f, 1.8f, 0.17f);
    drawBox(-0.02f, 1.8f, -0.02f, 0.02f, 1.95f, 0.02f);
    glc(Col(0.15f, 0.15f, 0.17f));
    drawBox(-0.2f, 1.1f, 0.17f, 0.2f, 1.7f, 0.18f);
    glc(Col(0.75f, 0.75f, 0.72f));
    drawBox(-0.08f, 1.25f, 0.18f, 0.08f, 1.45f, 0.19f);
    glc(Col(0.05f, 0.05f, 0.05f));
    drawBox(-0.27f, 1.2f, 0.18f, -0.17f, 1.65f, 0.26f);
    glc(Col(0.1f, 0.25f, 0.6f));
    drawBox(-0.32f, 1.95f, -0.04f, 0.32f, 2.2f, 0.04f);
    unlitOn();
    glColor3f(1, 1, 1);
    for (int s = 0; s < 2; ++s) {
        glPushMatrix();
        glRotatef(s * 180.0f, 0, 1, 0);
        glTranslatef(0, 2.075f, 0.045f);
        drawStrokeCentered("PHONE", 0.13f, 0.55f, 0.003f);
        glPopMatrix();
    }
    unlitOff();
    glPopMatrix();
}
static void drawTrash(float x, float z) {
    V3 b(x, CURB_H, z);
    glc(Col(0.12f, 0.25f, 0.16f));
    drawCyl(b, b + V3(0, 0.9f, 0), 0.24f, 0.3f, 12, false);
    drawCyl(b + V3(0, 0.86f, 0), b + V3(0, 0.93f, 0), 0.31f, 0.31f, 12, false);
    glc(Col(0.85f, 0.83f, 0.78f));
    drawSphere(b + V3(0.05f, 0.88f, 0), 0.26f, 0.12f, 0.24f);
    glc(Col(0.55f, 0.4f, 0.25f));
    drawSphere(b + V3(-0.1f, 0.94f, 0.08f), 0.12f, 0.08f, 0.1f);
}
static void drawBarricade(float x, float z, float rot) {
    glPushMatrix();
    glTranslatef(x, baseGround(x, z), z);
    glRotatef(degf(rot), 0, 1, 0);
    glc(Col(0.1f, 0.2f, 0.55f));
    drawBox(-1.2f, 0.75f, -0.04f, 1.2f, 1.0f, 0.04f);
    for (float sx : {-1.0f, 1.0f}) for (float fz : {-0.35f, 0.35f}) drawCyl(V3(sx, 0.95f, 0), V3(sx * 1.05f, 0.0f, fz), 0.04f, 0.04f, 5);
    unlitOn();
    glColor3f(1, 1, 1);
    for (int s = 0; s < 2; ++s) {
        glPushMatrix();
        glRotatef(s * 180.0f, 0, 1, 0);
        glTranslatef(0, 0.875f, 0.045f);
        drawStrokeCentered("POLICE LINE - DO NOT CROSS", 0.11f, 2.25f, 0.003f);
        glPopMatrix();
    }
    unlitOff();
    glPopMatrix();
}
static void drawPromLamp(float x, float z) {
    V3 b(x, CURB_H, z);
    glc(Col(0.08f, 0.09f, 0.08f));
    drawCyl(b, b + V3(0, 0.9f, 0), 0.2f, 0.12f, 10);
    drawCyl(b + V3(0, 0.9f, 0), b + V3(0, 4.2f, 0), 0.08f, 0.06f, 8);
    drawCyl(b + V3(0, 4.2f, 0), b + V3(0, 4.35f, 0), 0.16f, 0.12f, 8);
    unlitOn();
    glColor3f(0.98f, 0.95f, 0.85f);
    drawSphere(b + V3(0, 4.62f, 0), 0.3f);
    unlitOff();
}
static void drawParkBench(float x, float z) {
    glc(Col(0.1f, 0.1f, 0.1f));
    for (float lx : {-0.85f, 0.85f}) {
        drawBox(x + lx - 0.04f, CURB_H, z - 0.3f, x + lx + 0.04f, CURB_H + 0.45f, z + 0.28f);
        drawBox(x + lx - 0.04f, CURB_H + 0.45f, z - 0.3f, x + lx + 0.04f, CURB_H + 0.95f, z - 0.22f);
    }
    glc(Col(0.2f, 0.4f, 0.24f));
    for (int i = 0; i < 4; ++i) drawBox(x - 1.0f, CURB_H + 0.43f, z - 0.25f + i * 0.13f, x + 1.0f, CURB_H + 0.47f, z - 0.15f + i * 0.13f);
    for (int i = 0; i < 3; ++i) drawBox(x - 1.0f, CURB_H + 0.58f + i * 0.13f, z - 0.3f, x + 1.0f, CURB_H + 0.68f + i * 0.13f, z - 0.26f);
}
static void drawHotdogCart(float x, float z) {
    glPushMatrix();
    glTranslatef(x, CURB_H, z);
    glc(Col(0.1f, 0.1f, 0.1f));
    for (float wx : {-0.6f, 0.6f}) for (float wz : {-0.53f, 0.45f}) drawCyl(V3(wx, 0.28f, wz), V3(wx, 0.28f, wz + 0.08f), 0.28f, 0.28f, 10);
    glc(Col(0.78f, 0.79f, 0.82f));
    drawBox(-1.0f, 0.4f, -0.45f, 1.0f, 1.1f, 0.45f);
    glc(Col(0.6f, 0.6f, 0.62f));
    drawBox(-1.02f, 1.1f, -0.47f, 1.02f, 1.16f, 0.47f);
    for (int s = 0; s < 2; ++s) {
        glPushMatrix();
        glRotatef(s * 180.0f, 0, 1, 0);
        glc(Col(0.95f, 0.95f, 0.92f));
        drawBox(-0.9f, 0.55f, 0.45f, 0.9f, 1.02f, 0.47f);
        unlitOn();
        glColor3f(0.8f, 0.1f, 0.1f);
        glPushMatrix(); glTranslatef(0, 0.88f, 0.475f); drawStrokeCentered("HOT DOGS", 0.15f, 1.6f, 0.004f); glPopMatrix();
        glColor3f(0.1f, 0.25f, 0.65f);
        glPushMatrix(); glTranslatef(0, 0.66f, 0.475f); drawStrokeCentered("PRETZELS - SODA", 0.09f, 1.6f, 0.002f); glPopMatrix();
        unlitOff();
        glPopMatrix();
    }
    glc(Col(0.6f, 0.6f, 0.6f));
    drawCyl(V3(0, 1.16f, 0), V3(0, 2.65f, 0), 0.025f, 0.025f, 5, false);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 12; ++i) {
        float a0 = 2 * PI * i / 12, a1 = 2 * PI * (i + 1) / 12;
        if (i & 1) glColor3f(0.95f, 0.8f, 0.1f); else glColor3f(0.1f, 0.35f, 0.75f);
        V3 p0(cosf(a0) * 1.3f, 2.25f, sinf(a0) * 1.3f), p1(cosf(a1) * 1.3f, 2.25f, sinf(a1) * 1.3f), ap(0, 2.7f, 0);
        nrm(normalize(cross(p1 - ap, p0 - ap)));
        vtx(ap); vtx(p0); vtx(p1);
    }
    glEnd();
    glPopMatrix();
}
static void drawDumpster(float x, float z) {
    glc(Col(0.14f, 0.34f, 0.2f));
    drawBox(x - 1.45f, 0.18f, z - 0.95f, x + 1.45f, 1.22f, z + 0.95f);
    glc(Col(0.1f, 0.26f, 0.15f));
    drawBox(x - 1.5f, 0.6f, z - 0.98f, x + 1.5f, 0.75f, z + 0.98f);
    glc(Col(0.08f, 0.08f, 0.08f));
    drawBox(x - 1.5f, 1.22f, z - 1.0f, x + 1.5f, 1.3f, z + 1.0f);
    for (float wx : {-1.2f, 1.2f}) for (float wz : {-0.7f, 0.7f}) drawSphere(V3(x + wx, 0.12f + CURB_H * 0.5f, z + wz), 0.1f);
    unlitOn();
    glColor3f(1.0f, 0.4f, 0.8f);
    glPushMatrix();
    glTranslatef(x - 1.47f, 0.92f, z);
    glRotatef(-90, 0, 1, 0);
    drawStrokeCentered("SK8", 0.4f, 1.6f, 0.012f);
    glPopMatrix();
    unlitOff();
}

static void drawProps() {
    for (const Prop& p : g_props) {
        texOff();
        float facing = p.z < 0 ? PI : 0.0f;  // street furniture faces the walkers, away from the road
        switch (p.type) {
            case P_LAMP: drawLamp(p.x, p.z, p.rot, fabsf(p.x) >= 40.0f); break;
            case P_TREE:
                glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -2);
                glc(Col(0.22f, 0.17f, 0.12f));
                drawGround(p.x - 0.65f, p.z - 0.65f, p.x + 0.65f, p.z + 0.65f, CURB_H, 1, 2);
                glDisable(GL_POLYGON_OFFSET_FILL);
                drawTree(p.x, p.z, CURB_H, (int)(p.x * 10));
                break;
            case P_PLANTER_TREE:
                glc(Col(0.22f, 0.17f, 0.12f));
                drawBox(p.x - 1.8f, 0.68f, p.z - 0.8f, p.x + 1.8f, 0.71f, p.z + 0.8f);
                drawTree(p.x, p.z, 0.7f, (int)(p.x * 10) + 3);
                break;
            case P_HYDRANT: drawHydrant(p.x, p.z, p.z < 0 ? 0.0f : PI); break;
            case P_NEWSBOX: drawNewsbox(p.x, p.z, facing, p.variant); break;
            case P_MAILBOX: drawMailbox(p.x, p.z, facing); break;
            case P_PAYPHONE: drawPayphone(p.x, p.z, facing); break;
            case P_TRASH: drawTrash(p.x, p.z); break;
            case P_BARRICADE: drawBarricade(p.x, p.z, p.rot); break;
            case P_PARKED_CAR:
                glPushMatrix();
                glTranslatef(p.x, 0, p.z);
                glRotatef(180, 0, 1, 0);
                glCallList(g_carLists[(p.variant * 3 + 1) & 7]);
                glPopMatrix();
                break;
            case P_PROM_LAMP: drawPromLamp(p.x, p.z); break;
            case P_BENCH_WOOD: drawParkBench(p.x, p.z); break;
            case P_HOTDOG: drawHotdogCart(p.x, p.z); break;
            case P_MANHOLE:
                glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -2);
                glc(Col(0.16f, 0.15f, 0.14f)); drawDisc(p.x, 0.0f, p.z, 0.42f, 0.42f);
                glc(Col(0.24f, 0.23f, 0.21f)); drawDisc(p.x, 0.002f, p.z, 0.3f, 0.3f);
                glDisable(GL_POLYGON_OFFSET_FILL);
                break;
            case P_DUMPSTER: drawDumpster(p.x, p.z); break;
            default: break;
        }
    }
}

// Pipe split into short segments so per-vertex fog stays smooth.
static void drawTube(const V3& a, const V3& b, float r, int seg = 8) {
    int n = std::max(1, (int)(len(b - a) / 4.0f));
    for (int i = 0; i < n; ++i) drawCyl(lerpv(a, b, (float)i / n), lerpv(a, b, (float)(i + 1) / n), r, r, seg, false);
}
static void drawRails() {
    texOff();
    for (const Rail& r : g_rails) {
        if (r.drawTube) {
            glc(Col(0.58f, 0.6f, 0.63f));
            drawTube(r.a, r.b, 0.035f);
            int n = railPostCount(r);
            for (int k = 0; k <= n; ++k) {
                V3 p = lerpv(r.a, r.b, (float)k / n);
                float base = queryGround(p.x, p.z, p.y - 0.1f).h;
                drawCyl(V3(p.x, base, p.z), V3(p.x, p.y, p.z), 0.03f, 0.03f, 6, false);
            }
        } else if (r.kind == RK_LEDGE) {
            glc(Col(0.5f, 0.52f, 0.55f));
            drawTube(r.a + V3(0, -0.015f, 0), r.b + V3(0, -0.015f, 0), 0.022f, 5);
        }
    }
}
static void drawRailing(float x0, float z0, float x1, float z1, float y0, float y1, float pitch, const Col& c, float barW) {
    glc(c);
    V3 a(x0, y1, z0), b(x1, y1, z1);
    drawTube(a, b, 0.03f, 6);
    drawTube(V3(x0, y0 + 0.15f, z0), V3(x1, y0 + 0.15f, z1), 0.02f, 5);
    float L = len2D(x1 - x0, z1 - z0);
    int n = std::max(1, (int)(L / pitch));
    for (int i = 0; i <= n; ++i) {
        float t = (float)i / n, x = lerpf(x0, x1, t), z = lerpf(z0, z1, t);
        drawBox(x - barW, y0, z - barW, x + barW, y1, z + barW);
    }
}
static void drawSubway() {
    const float x0 = -25.94f, x1 = -22.0f, zN = 9.06f, zS = 11.54f;
    texOn(g_texConcrete);
    glc(Col(0.55f, 0.55f, 0.55f));
    for (int i = 0; i < 8; ++i)
        drawBox(x1 - (i + 1) * 0.4925f, -3.0f, zN, x1 - i * 0.4925f, CURB_H - (i + 1) * 0.36f, zS);
    texOff();
    glc(Col(0.85f, 0.86f, 0.82f));
    wallStripZ(zN, 1, x0, x1, -3.0f, CURB_H);
    wallStripZ(zS, -1, x0, x1, -3.0f, CURB_H);
    glNormal3f(1, 0, 0);
    glBegin(GL_QUADS);
    glVertex3f(x0, -3.0f, zN); glVertex3f(x0, -3.0f, zS); glVertex3f(x0, CURB_H, zS); glVertex3f(x0, CURB_H, zN);
    glEnd();
    const Col green(0.1f, 0.3f, 0.18f);
    drawRailing(-26.0f, 9.0f, -22.0f, 9.0f, CURB_H, 1.1f, 0.15f, green, 0.012f);
    drawRailing(-26.0f, 11.6f, -22.0f, 11.6f, CURB_H, 1.1f, 0.15f, green, 0.012f);
    drawRailing(-26.0f, 9.0f, -26.0f, 11.6f, CURB_H, 1.1f, 0.15f, green, 0.012f);
    for (float z : {9.0f, 11.6f}) {
        glc(green);
        drawCyl(V3(-26.0f, CURB_H, z), V3(-26.0f, 2.45f, z), 0.05f, 0.04f, 8);
        unlitOn();
        glColor3f(0.35f, 0.95f, 0.45f);
        drawSphere(V3(-26.0f, 2.62f, z), 0.18f);
        unlitOff();
    }
    glc(Col(0.05f, 0.05f, 0.05f));
    drawBox(-25.2f, 0.45f, 8.92f, -22.8f, 0.95f, 8.94f);
    unlitOn();
    glColor3f(1, 1, 1);
    glPushMatrix();
    glTranslatef(-24.0f, 0.7f, 8.915f);
    glRotatef(180, 0, 1, 0);
    drawStrokeCentered("SUBWAY", 0.24f, 2.2f, 0.006f);
    glPopMatrix();
    unlitOff();
}
static void drawFountainStatic() {
    const float cx = FOUNTAIN_C.x, cz = FOUNTAIN_C.z;
    texOn(g_texGranite);
    glc(C_GRANITE);
    drawBox(cx - 0.6f, CURB_H, cz - 0.6f, cx + 0.6f, 1.0f, cz + 0.6f);
    texOff();
    glc(C_GRANITE * 0.9f);
    drawCyl(V3(cx, 1.0f, cz), V3(cx, 1.35f, cz), 0.45f, 1.1f, 18);
    drawCyl(V3(cx, 1.35f, cz), V3(cx, 1.65f, cz), 0.12f, 0.08f, 8);
}
static void drawScaffold() {
    const Building* b = nullptr;
    for (int i = 0; i < NORTH_ROW_N; ++i) if (NORTH_ROW[i].x0 == SCAFFOLD_X0) b = &NORTH_ROW[i];
    const Col pipe(0.25f, 0.3f, 0.45f), ply(0.18f, 0.32f, 0.22f);
    glc(pipe);
    for (float x = SCAFFOLD_X0; x <= SCAFFOLD_X1 + 0.01f; x += 3.0f) {
        drawCyl(V3(x, CURB_H, -9.3f), V3(x, 3.4f, -9.3f), 0.05f, 0.05f, 6, false);
        drawCyl(V3(x, CURB_H, -13.7f), V3(x, 3.4f, -13.7f), 0.05f, 0.05f, 6, false);
        drawCyl(V3(x, 3.6f, -13.8f), V3(x, 12.0f, -13.8f), 0.04f, 0.04f, 6, false);
    }
    for (float y = 6.0f; y <= 12.0f; y += 3.0f) drawTube(V3(SCAFFOLD_X0, y, -13.8f), V3(SCAFFOLD_X1, y, -13.8f), 0.04f, 6);
    glc(Col(0.3f, 0.3f, 0.32f));
    drawBox(SCAFFOLD_X0 - 0.2f, 3.2f, -9.42f, SCAFFOLD_X1 + 0.2f, 3.4f, -9.18f);
    texOn(g_texWood);
    glc(ply);
    drawBox(SCAFFOLD_X0 - 0.2f, 3.4f, -13.95f, SCAFFOLD_X1 + 0.2f, 3.6f, -9.1f, 1.2f, 1.2f);
    drawBox(SCAFFOLD_X0 - 0.2f, 3.6f, -9.2f, SCAFFOLD_X1 + 0.2f, 4.5f, -9.1f, 1.2f, 1.2f);
    texOff();
    unlitOn();
    glColor3f(1.0f, 0.97f, 0.85f);
    for (float x = SCAFFOLD_X0 + 1.5f; x < SCAFFOLD_X1; x += 3.0f) drawBox(x - 0.3f, 3.36f, -11.6f, x + 0.3f, 3.4f, -11.4f);
    glColor3f(0.92f, 0.92f, 0.9f);
    for (float x : {SCAFFOLD_X0 + 2.0f, SCAFFOLD_X1 - 2.0f}) {
        glPushMatrix(); glTranslatef(x, 4.05f, -9.09f); drawStrokeCentered("POST NO BILLS", 0.2f, 3.0f, 0.004f); glPopMatrix();
    }
    if (b) {
        float cx = (SCAFFOLD_X0 + SCAFFOLD_X1) * 0.5f;
        glc(b->signBg);
        drawBox(cx - 3.5f, 3.65f, -9.09f, cx + 3.5f, 4.45f, -9.06f);
        glc(b->signFg);
        glLineWidth(2.0f);
        glPushMatrix(); glTranslatef(cx, 4.05f, -9.05f); drawStrokeCentered(b->sign, 0.42f, 6.4f, 0.01f); glPopMatrix();
        glLineWidth(1.0f);
    }
    unlitOff();
}

// ---- Static shadows (ground-projected, one stencil pass per frame) -------------------
struct P2 { float x, z; };
static std::vector<P2> hull2D(std::vector<P2> p) {
    if (p.size() < 3) return p;
    std::sort(p.begin(), p.end(), [](const P2& a, const P2& b) { return a.x < b.x || (a.x == b.x && a.z < b.z); });
    auto cr = [](const P2& o, const P2& a, const P2& b) { return (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x); };
    std::vector<P2> h(p.size() * 2);
    size_t k = 0;
    for (size_t i = 0; i < p.size(); ++i) { while (k >= 2 && cr(h[k - 2], h[k - 1], p[i]) <= 0) --k; h[k++] = p[i]; }
    for (size_t i = p.size() - 1, t = k + 1; i > 0; --i) { while (k >= t && cr(h[k - 2], h[k - 1], p[i - 1]) <= 0) --k; h[k++] = p[i - 1]; }
    h.resize(k - 1);
    return h;
}
static std::vector<P2> clipZ(const std::vector<P2>& poly, float c, bool keepGreater) {
    std::vector<P2> out;
    for (size_t i = 0; i < poly.size(); ++i) {
        const P2& a = poly[i];
        const P2& b = poly[(i + 1) % poly.size()];
        bool ia = keepGreater ? a.z >= c : a.z <= c, ib = keepGreater ? b.z >= c : b.z <= c;
        if (ia) out.push_back(a);
        if (ia != ib) {
            float t = (c - a.z) / (b.z - a.z);
            out.push_back(P2{a.x + (b.x - a.x) * t, c});
        }
    }
    return out;
}
static void fillPoly(const std::vector<P2>& poly, float y) {
    if (poly.size() < 3) return;
    glBegin(GL_POLYGON);
    for (const P2& q : poly) glVertex3f(q.x, y, q.z);
    glEnd();
}
static std::vector<P2> projectHull(const std::vector<V3>& pts, float h) {
    std::vector<P2> q;
    for (const V3& v : pts) {
        float t = (v.y - h) / SUN_DIR.y;
        q.push_back(P2{v.x - SUN_DIR.x * t, v.z - SUN_DIR.z * t});
    }
    return hull2D(q);
}
// Shadow of a convex caster.  Near a curb the shadow is split between sidewalk and street levels.
static void castShadow(const std::vector<V3>& pts, float baseH, float cz) {
    bool south = cz > SCURB_Z - 3 && cz < SCURB_Z + 4, north = cz > NCURB_Z - 4 && cz < NCURB_Z + 3;
    if (south || north) {
        float c = south ? SCURB_Z : NCURB_Z;
        fillPoly(clipZ(projectHull(pts, CURB_H), c, south), CURB_H + 0.012f);
        fillPoly(clipZ(projectHull(pts, 0.0f), c, !south), 0.012f);
    } else {
        fillPoly(projectHull(pts, baseH), baseH + 0.012f);
    }
}
static void boxPoints(std::vector<V3>& pts, float x0, float y0, float z0, float x1, float y1, float z1) {
    for (int i = 0; i < 8; ++i) pts.push_back(V3((i & 1) ? x1 : x0, (i & 2) ? y1 : y0, (i & 4) ? z1 : z0));
}
static void ringPoints(std::vector<V3>& pts, const V3& c, float r, float y0, float y1, int n = 10) {
    for (int i = 0; i < n; ++i) {
        float a = 2 * PI * i / n;
        pts.push_back(V3(c.x + cosf(a) * r, y0, c.z + sinf(a) * r));
        pts.push_back(V3(c.x + cosf(a) * r, y1, c.z + sinf(a) * r));
    }
}
static void drawStaticShadows() {
    std::vector<V3> pts;
    for (size_t i = 0; i < g_solids.size(); ++i) {
        const Solid& s = g_solids[i];
        if (s.y0 < 0) continue;  // buildings, boundaries
        float w = s.x1 - s.x0, d = s.z1 - s.z0;
        if (std::min(w, d) < 0.25f && std::max(w, d) > 2.5f) continue;  // see-through railings
        if (std::find(g_carRoofSolids.begin(), g_carRoofSolids.end(), (int)i) != g_carRoofSolids.end()) continue;
        float cx = (s.x0 + s.x1) * 0.5f, cz = (s.z0 + s.z1) * 0.5f;
        float base = queryGround(cx, cz, s.yLow - 0.05f).h;
        pts.clear();
        const float xs[4] = {s.x0, s.x0, s.x1, s.x1}, zs[4] = {s.z0, s.z1, s.z1, s.z0};
        for (int k = 0; k < 4; ++k) {
            pts.push_back(V3(xs[k], base, zs[k]));
            pts.push_back(V3(xs[k], solidTop(s, xs[k], zs[k]), zs[k]));
        }
        castShadow(pts, base, cz);
    }
    for (int i = 0; i < PARKED_N; ++i) {
        float cx = PARKED_X[i];
        pts.clear();
        boxPoints(pts, cx - 2.3f, 0.0f, PARK_Z - 0.9f, cx + 2.3f, 0.95f, PARK_Z + 0.9f);
        boxPoints(pts, cx - 1.1f, 0.95f, PARK_Z - 0.8f, cx + 1.0f, 1.45f, PARK_Z + 0.8f);
        castShadow(pts, 0.0f, PARK_Z);
    }
    for (const Prop& p : g_props) {
        pts.clear();
        if (p.type == P_TREE || p.type == P_PLANTER_TREE) {
            float base = p.type == P_TREE ? CURB_H : 0.7f;
            ringPoints(pts, V3(p.x, 0, p.z), 1.7f, base + 2.9f, base + 4.3f);
            castShadow(pts, CURB_H, p.z);
        } else if (p.type == P_LAMP) {
            V3 h = V3(p.x, CURB_H + 6.75f, p.z) + fwdFromYaw(p.rot) * 2.3f;
            boxPoints(pts, h.x - 0.25f, h.y - 0.1f, h.z - 0.5f, h.x + 0.25f, h.y + 0.1f, h.z + 0.5f);
            castShadow(pts, CURB_H, p.z);
        } else if (p.type == P_HOTDOG) {
            ringPoints(pts, V3(p.x, 0, p.z), 1.3f, CURB_H + 2.25f, CURB_H + 2.7f);
            castShadow(pts, CURB_H, p.z);
        }
    }
    for (const Rail& r : g_rails) {
        if (!r.drawTube || fabsf(r.a.y - r.b.y) > 0.1f) continue;
        V3 dir = normalize(r.b - r.a), side(dir.z * 0.04f, 0, -dir.x * 0.04f);
        pts.clear();
        pts.push_back(r.a + side); pts.push_back(r.a - side); pts.push_back(r.b + side); pts.push_back(r.b - side);
        castShadow(pts, queryGround((r.a.x + r.b.x) * 0.5f, (r.a.z + r.b.z) * 0.5f, r.a.y - 0.1f).h, r.a.z);
    }
    pts.clear();  // sidewalk shed deck
    boxPoints(pts, SCAFFOLD_X0 - 0.2f, 3.4f, -13.95f, SCAFFOLD_X1 + 0.2f, 4.5f, -9.1f);
    castShadow(pts, CURB_H, -11.5f);
}

static GLuint g_worldList = 0, g_shadowList = 0;
static void compileStatic() {
    buildSphereList();
    for (int i = 0; i < 8; ++i) {
        g_carLists[i] = glGenLists(1);
        glNewList(g_carLists[i], GL_COMPILE);
        drawCarModel(i);
        glEndList();
    }
    g_worldList = glGenLists(1);
    glNewList(g_worldList, GL_COMPILE);
    drawGroundAll();
    drawMarkings();
    drawBuildings();
    for (const Solid& s : g_solids) if (s.render) drawSolid(s);
    texOff();
    drawRails();
    drawProps();
    drawSubway();
    drawFountainStatic();
    drawScaffold();
    drawRailing(-EXT, 43.65f, EXT, 43.65f, CURB_H, 1.15f, 2.0f, Col(0.12f, 0.16f, 0.14f), 0.03f);
    drawSkyline();
    drawBridge();
    texOff();
    glEndList();
    g_shadowList = glGenLists(1);
    glNewList(g_shadowList, GL_COMPILE);
    drawStaticShadows();
    glEndList();
}

// ----------------------------------------------------------------------------
//  Dynamic world: particles, water, traffic, pedestrians, pigeons
// ----------------------------------------------------------------------------
static float g_time = 0.0f;

enum PartKind { PK_WATER, PK_SPARK, PK_DUST, PK_STEAM };
struct Particle {
    V3 p, v;
    float life, maxLife, size, grow, gravity, drag, floorY, alpha;
    Col c;
    int kind;
    bool ripple;  // leaves a ripple where it lands
};
static std::vector<Particle> g_parts;
static const size_t MAX_PARTS = 3000;

struct Ripple { float x, y, z, r, life, maxLife, speed; };
static std::vector<Ripple> g_ripples;
static void addRipple(float x, float y, float z, float speed = 0.6f, float life = 1.3f) {
    if (g_ripples.size() < 260) g_ripples.push_back({x, y, z, 0.03f, life, life, speed});
}
// Returned pointer is valid until the next emit (storage is reserved up front).
static Particle* emit(int kind, const V3& p, const V3& v, float life, float size, const Col& c, float alpha) {
    Particle q;
    q.p = p; q.v = v; q.life = q.maxLife = life; q.size = size; q.grow = 0; q.alpha = alpha; q.c = c;
    q.kind = kind; q.ripple = false; q.floorY = -50.0f;
    q.gravity = kind == PK_STEAM ? -0.35f : (kind == PK_DUST ? 0.6f : 9.8f);
    q.drag = kind == PK_DUST ? 3.0f : (kind == PK_STEAM ? 0.9f : 0.25f);
    if (g_parts.size() < MAX_PARTS) { g_parts.push_back(q); return &g_parts.back(); }
    Particle& slot = g_parts[xrand() % g_parts.size()];
    slot = q;
    return &slot;
}
static void emitSplash(const V3& p, int n, float power, float floorY) {
    for (int i = 0; i < n; ++i) {
        float a = frand() * 2 * PI, s = frange(0.3f, 1.0f) * power;
        Particle* q = emit(PK_WATER, p, V3(cosf(a) * s, frange(1.5f, 3.2f) * (0.5f + 0.5f * power), sinf(a) * s),
                           1.2f, frange(0.05f, 0.11f), Col(0.85f, 0.92f, 1.0f), 0.7f);
        q->floorY = floorY;
    }
    addRipple(p.x, floorY + 0.01f, p.z, 1.2f, 1.0f);
}
static void emitSparks(const V3& p, const V3& dir, int n) {
    for (int i = 0; i < n; ++i) {
        V3 v = dir * frange(-2.5f, -0.5f) + V3(frange(-1.2f, 1.2f), frange(0.5f, 2.5f), frange(-1.2f, 1.2f));
        Particle* q = emit(PK_SPARK, p, v, frange(0.2f, 0.45f), frange(0.03f, 0.06f), Col(1.0f, frange(0.55f, 0.85f), 0.25f), 1.0f);
        q->floorY = p.y - 1.5f;
    }
}
static void emitDust(const V3& p, int n, float spread) {
    for (int i = 0; i < n; ++i) {
        float a = frand() * 2 * PI, s = frange(0.5f, 1.6f) * spread;
        Particle* q = emit(PK_DUST, p + V3(0, 0.05f, 0), V3(cosf(a) * s, frange(0.1f, 0.6f), sinf(a) * s), frange(0.5f, 0.9f),
                           frange(0.12f, 0.22f), Col(0.75f, 0.72f, 0.68f), 0.35f);
        q->grow = 0.5f;
    }
}

static void updateParticles(float dt) {
    for (size_t i = 0; i < g_parts.size();) {
        Particle& q = g_parts[i];
        q.life -= dt;
        q.v.y -= q.gravity * dt;
        q.v *= 1.0f / (1.0f + q.drag * dt);
        q.p += q.v * dt;
        q.size += q.grow * dt;
        bool dead = q.life <= 0;
        if (!dead && q.p.y < q.floorY) {
            dead = true;
            if (q.ripple) addRipple(q.p.x, q.floorY + 0.01f, q.p.z, 0.45f, 1.1f);
        }
        if (dead) { q = g_parts.back(); g_parts.pop_back(); }
        else ++i;
    }
    for (size_t i = 0; i < g_ripples.size();) {
        Ripple& r = g_ripples[i];
        r.life -= dt;
        r.r += r.speed * dt;
        if (r.life <= 0) { r = g_ripples.back(); g_ripples.pop_back(); }
        else ++i;
    }
}
static void drawParticles(const V3& camRight, const V3& camUp) {
    if (g_parts.empty()) return;
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    texOn(g_texParticle);
    for (int pass = 0; pass < 2; ++pass) {
        glBlendFunc(GL_SRC_ALPHA, pass == 0 ? GL_ONE_MINUS_SRC_ALPHA : GL_ONE);
        glBegin(GL_QUADS);
        for (const Particle& q : g_parts) {
            if ((q.kind == PK_SPARK) != (pass == 1)) continue;
            float t = q.life / q.maxLife;
            float a = q.alpha * (q.kind == PK_STEAM ? sinf(t * PI) : std::min(1.0f, t * 2.5f));
            glColor4f(q.c.r, q.c.g, q.c.b, a);
            V3 r = camRight * q.size, u = camUp * q.size;
            glTexCoord2f(0, 0); vtx(q.p - r - u);
            glTexCoord2f(1, 0); vtx(q.p + r - u);
            glTexCoord2f(1, 1); vtx(q.p + r + u);
            glTexCoord2f(0, 1); vtx(q.p - r + u);
        }
        glEnd();
    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    texOff();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}
static void drawRipples() {
    glBegin(GL_QUADS);
    for (const Ripple& r : g_ripples) {
        float t = r.life / r.maxLife, w = 0.035f + r.r * 0.05f;
        glColor4f(0.92f, 0.96f, 1.0f, 0.5f * t);
        for (int i = 0; i < 20; ++i) {
            float a0 = 2 * PI * i / 20, a1 = 2 * PI * (i + 1) / 20;
            float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
            glVertex3f(r.x + c0 * r.r, r.y, r.z + s0 * r.r);
            glVertex3f(r.x + c0 * (r.r + w), r.y, r.z + s0 * (r.r + w));
            glVertex3f(r.x + c1 * (r.r + w), r.y, r.z + s1 * (r.r + w));
            glVertex3f(r.x + c1 * r.r, r.y, r.z + s1 * r.r);
        }
    }
    glEnd();
}

// ---- Ambient water emitters (fountain, open hydrant, manhole steam) ----------
static const float FOUNTAIN_WATER_Y = 0.5f;
static float g_accJet = 0, g_accArc = 0, g_accSpill = 0, g_accHydrant = 0, g_accSteam = 0, g_accDrip = 0;
static void updateEmitters(float dt) {
    const V3 fc = FOUNTAIN_C;
    for (g_accJet += dt * 70.0f; g_accJet >= 1; g_accJet -= 1) {  // central jet
        float a = frand() * 2 * PI, s = frand() < 0.3f ? frange(0.05f, 0.5f) : frange(1.0f, 1.6f);
        Particle* q = emit(PK_WATER, V3(fc.x, 1.68f, fc.z), V3(cosf(a) * s, frange(6.2f, 7.2f), sinf(a) * s), 3.0f,
                           frange(0.05f, 0.1f), Col(0.86f, 0.93f, 1.0f), 0.55f);
        q->floorY = s < 0.6f ? 1.35f : FOUNTAIN_WATER_Y;
        q->ripple = s >= 0.6f && frand() < 0.35f;
    }
    for (g_accArc += dt * 64.0f; g_accArc >= 1; g_accArc -= 1) {  // eight arcing jets from the rim
        int k = (int)(xrand() % 8);
        float a = k * PI / 4, cx = cosf(a), cz = sinf(a);
        Particle* q = emit(PK_WATER, V3(fc.x + cx * 3.2f, 0.58f, fc.z + cz * 3.2f),
                           V3(-cx * frange(2.0f, 2.4f), frange(2.8f, 3.2f), -cz * frange(2.0f, 2.4f)), 1.5f, 0.05f,
                           Col(0.86f, 0.93f, 1.0f), 0.5f);
        q->floorY = FOUNTAIN_WATER_Y;
        q->ripple = frand() < 0.3f;
    }
    for (g_accSpill += dt * 40.0f; g_accSpill >= 1; g_accSpill -= 1) {  // curtain spilling off the bowl
        float a = frand() * 2 * PI;
        Particle* q = emit(PK_WATER, V3(fc.x + cosf(a) * 1.12f, 1.34f, fc.z + sinf(a) * 1.12f),
                           V3(cosf(a) * 0.35f, 0, sinf(a) * 0.35f), 1.0f, 0.06f, Col(0.86f, 0.93f, 1.0f), 0.45f);
        q->floorY = FOUNTAIN_WATER_Y;
        q->ripple = frand() < 0.15f;
    }
    for (g_accHydrant += dt * 75.0f; g_accHydrant >= 1; g_accHydrant -= 1) {  // open hydrant, summer style
        Particle* q = emit(PK_WATER, V3(HYDRANT_POS.x, 0.6f, HYDRANT_POS.z + 0.25f),
                           V3(frange(-0.7f, 0.7f), frange(2.4f, 3.8f), frange(5.0f, 7.5f)), 2.0f, frange(0.06f, 0.13f),
                           Col(0.88f, 0.94f, 1.0f), 0.6f);
        q->floorY = 0.0f;
        q->ripple = frand() < 0.25f;
        if (frand() < 0.15f) {
            Particle* m = emit(PK_STEAM, V3(HYDRANT_POS.x + frange(-0.5f, 0.5f), 0.2f, frange(-6.5f, -4.0f)),
                               V3(frange(-0.2f, 0.2f), 0.3f, 0), 1.2f, 0.3f, Col(0.92f, 0.95f, 1.0f), 0.18f);
            m->grow = 0.6f;
        }
    }
    for (g_accSteam += dt * 9.0f; g_accSteam >= 1; g_accSteam -= 1) {
        const Prop* mh = nullptr;
        int pick = (int)(xrand() % 3), seen = 0;
        for (const Prop& p : g_props) if (p.type == P_MANHOLE && seen++ == pick) mh = &p;
        if (!mh) break;
        Particle* q = emit(PK_STEAM, V3(mh->x + frange(-0.25f, 0.25f), 0.05f, mh->z + frange(-0.25f, 0.25f)),
                           V3(0.35f + frange(-0.1f, 0.1f), frange(0.5f, 0.9f), frange(-0.1f, 0.1f)), frange(3.0f, 4.0f), 0.35f,
                           Col(0.93f, 0.93f, 0.93f), 0.32f);
        q->grow = 0.45f;
    }
    for (g_accDrip += dt * 2.5f; g_accDrip >= 1; g_accDrip -= 1) {  // rain-drip ripples in puddles
        const Puddle& pd = g_puddles[xrand() % g_puddles.size()];
        float a = frand() * 2 * PI, r = sqrtf(frand()) * 0.7f;
        float y = baseGround(pd.x, pd.z) + 0.012f;
        addRipple(pd.x + cosf(a) * r * pd.rx, y, pd.z + sinf(a) * r * pd.rz, 0.35f, 1.4f);
    }
}

// ---- Water shading -----------------------------------------------------------
static const Col SKY_HORIZON(0.86f, 0.80f, 0.70f), SKY_ZENITH(0.36f, 0.55f, 0.82f);
static const Col SUN_COL(1.0f, 0.88f, 0.68f);
static void waterColour(const V3& p, const V3& n, const V3& cam, const Col& deep, float& r, float& g, float& b) {
    V3 v = normalize(cam - p);
    float ndv = std::max(0.0f, dot(n, v));
    float fres = 0.05f + 0.95f * powf(1.0f - ndv, 5.0f);
    V3 h = normalize(SUN_DIR + v);
    float spec = powf(std::max(0.0f, dot(n, h)), 140.0f) * 2.2f;
    r = std::min(1.0f, lerpf(deep.r, 0.66f, fres) + SUN_COL.r * spec);
    g = std::min(1.0f, lerpf(deep.g, 0.74f, fres) + SUN_COL.g * spec);
    b = std::min(1.0f, lerpf(deep.b, 0.84f, fres) + SUN_COL.b * spec);
}
static std::vector<float> g_riverXs, g_riverZs;
static void buildRiverGrid() {
    for (float x = -EXT; x < -80.0f; x += 6.0f) g_riverXs.push_back(x);
    for (float x = -80.0f; x < 80.0f; x += 2.0f) g_riverXs.push_back(x);
    for (float x = 80.0f; x <= EXT + 0.1f; x += 6.0f) g_riverXs.push_back(x);
    for (float z = BULKHEAD_Z; z < 80.0f; z += 1.5f) g_riverZs.push_back(z);
    for (float z = 80.0f; z < 160.0f; z += 4.0f) g_riverZs.push_back(z);
    for (float z = 160.0f; z <= 304.0f; z += 12.0f) g_riverZs.push_back(z);
}
static void riverWave(float x, float z, float t, float& h, V3& n) {
    float d = (fabsf(x) < 80.0f && z < 90.0f) ? 1.0f : 0.0f;  // fine ripples only where the grid is dense
    float a = x * 0.21f + z * 0.13f + t * 1.1f, b = -x * 0.13f + z * 0.37f + t * 1.7f, c = x * 0.71f + z * 0.53f + t * 2.6f;
    h = WATER_Y + 0.1f * sinf(a) + 0.06f * sinf(b) + d * 0.035f * sinf(c);
    float dx = 0.021f * cosf(a) - 0.0078f * cosf(b) + d * 0.02485f * cosf(c);
    float dz = 0.013f * cosf(a) + 0.0222f * cosf(b) + d * 0.01855f * cosf(c);
    n = normalize(V3(-dx * 3.0f, 1.0f, -dz * 3.0f));  // exaggerated slope for livelier glints
}
static void drawRiver(const V3& cam) {
    const Col deep(0.07f, 0.18f, 0.24f);
    glDisable(GL_LIGHTING);
    std::vector<float> row0(g_riverXs.size() * 6), row1(g_riverXs.size() * 6);
    auto fillRow = [&](float z, std::vector<float>& out) {
        for (size_t i = 0; i < g_riverXs.size(); ++i) {
            float h; V3 n;
            riverWave(g_riverXs[i], z, g_time, h, n);
            float* o = &out[i * 6];
            o[0] = g_riverXs[i]; o[1] = h; o[2] = z;
            waterColour(V3(o[0], h, z), n, cam, deep, o[3], o[4], o[5]);
        }
    };
    fillRow(g_riverZs[0], row0);
    for (size_t j = 1; j < g_riverZs.size(); ++j) {
        fillRow(g_riverZs[j], row1);
        glBegin(GL_TRIANGLE_STRIP);
        for (size_t i = 0; i < g_riverXs.size(); ++i) {
            const float* a = &row0[i * 6];
            const float* b = &row1[i * 6];
            glColor3f(a[3], a[4], a[5]); glVertex3f(a[0], a[1], a[2]);
            glColor3f(b[3], b[4], b[5]); glVertex3f(b[0], b[1], b[2]);
        }
        glEnd();
        row0.swap(row1);
    }
    glEnable(GL_BLEND);  // foam lapping against the bulkhead
    glBegin(GL_QUAD_STRIP);
    for (float x = -EXT; x <= EXT; x += 4.0f) {
        float a = 0.25f + 0.2f * sinf(x * 0.4f + g_time * 2.0f);
        glColor4f(0.95f, 0.97f, 1.0f, a); glVertex3f(x, WATER_Y + 0.13f, BULKHEAD_Z + 0.02f);
        glColor4f(0.95f, 0.97f, 1.0f, 0.0f); glVertex3f(x, WATER_Y + 0.12f, BULKHEAD_Z + 1.4f);
    }
    glEnd();
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

static float g_ferryX = -150.0f;
static void drawFerry() {
    const float z = 150.0f, x = g_ferryX, y = WATER_Y + 0.08f * sinf(g_time * 0.9f);
    texOff();
    glc(Col(0.95f, 0.45f, 0.12f));
    drawBox(x - 18, y - 0.5f, z - 5, x + 18, y + 3.0f, z + 5);
    glc(Col(0.92f, 0.92f, 0.9f));
    drawBox(x - 15, y + 3.0f, z - 4.4f, x + 15, y + 5.6f, z + 4.4f);
    drawBox(x - 9, y + 5.6f, z - 3.5f, x + 9, y + 7.8f, z + 3.5f);
    glc(Col(0.2f, 0.25f, 0.3f));
    drawBox(x - 14.8f, y + 3.8f, z - 4.45f, x + 14.8f, y + 4.9f, z + 4.45f);
    glc(Col(0.95f, 0.45f, 0.12f));
    drawBox(x - 2, y + 7.8f, z - 1, x + 2, y + 9.0f, z + 1);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBegin(GL_TRIANGLES);  // wake
    glColor4f(1, 1, 1, 0.45f); glVertex3f(x - 18, WATER_Y + 0.15f, z - 5); glVertex3f(x - 18, WATER_Y + 0.15f, z + 5);
    glColor4f(1, 1, 1, 0.0f); glVertex3f(x - 70, WATER_Y + 0.15f, z);
    glColor4f(1, 1, 1, 0.5f); glVertex3f(x + 18, WATER_Y + 0.15f, z - 5);
    glColor4f(1, 1, 1, 0.0f); glVertex3f(x - 40, WATER_Y + 0.15f, z - 16); glVertex3f(x - 18, WATER_Y + 0.15f, z - 5);
    glColor4f(1, 1, 1, 0.5f); glVertex3f(x + 18, WATER_Y + 0.15f, z + 5);
    glColor4f(1, 1, 1, 0.0f); glVertex3f(x - 18, WATER_Y + 0.15f, z + 5); glVertex3f(x - 40, WATER_Y + 0.15f, z + 16);
    glEnd();
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// Transparent water drawn after opaque geometry: fountain pool, puddles, ripples.
static void drawWaterSurfaces(const V3& cam) {
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    const int N = 18;
    const float x0 = -3.6f, z0 = 22.4f, s = 7.2f / N;
    const Col teal(0.12f, 0.36f, 0.42f);
    for (int j = 0; j < N; ++j) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int i = 0; i <= N; ++i)
            for (int k = 0; k < 2; ++k) {
                float x = x0 + i * s, z = z0 + (j + k) * s;
                float dx = x - FOUNTAIN_C.x, dz = z - FOUNTAIN_C.z, rr = std::max(0.2f, len2D(dx, dz));
                float ph = rr * 5.0f - g_time * 4.0f, amp = 0.02f * expf(-rr * 0.25f);
                float h = FOUNTAIN_WATER_Y + amp * sinf(ph), dr = amp * 5.0f * cosf(ph);
                V3 n = normalize(V3(-dr * dx / rr * 4.0f, 1.0f, -dr * dz / rr * 4.0f));
                float r, g, b;
                waterColour(V3(x, h, z), n, cam, teal, r, g, b);
                glColor4f(r, g, b, 0.72f);
                glVertex3f(x, h, z);
            }
        glEnd();
    }
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2, -4);
    for (const Puddle& pd : g_puddles) {
        float y = baseGround(pd.x, pd.z) + 0.006f;
        float r, g, b;
        waterColour(V3(pd.x, y, pd.z), V3(0, 1, 0), cam, Col(0.16f, 0.18f, 0.2f), r, g, b);
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(r, g, b, 0.75f);
        glVertex3f(pd.x, y, pd.z);
        glColor4f(r, g, b, 0.0f);
        for (int i = 0; i <= 24; ++i) {
            float a = 2 * PI * i / 24, wob = 1.0f + 0.12f * sinf(a * 3 + pd.x);
            glVertex3f(pd.x + cosf(a) * pd.rx * wob, y, pd.z + sinf(a) * pd.rz * wob);
        }
        glEnd();
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
    drawRipples();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}
static bool inPuddle(float x, float z) {
    for (const Puddle& pd : g_puddles) {
        float dx = (x - pd.x) / pd.rx, dz = (z - pd.z) / pd.rz;
        if (dx * dx + dz * dz < 0.8f) return true;
    }
    return false;
}

// ---- Traffic -----------------------------------------------------------------
struct Car { float x, z, dir, speed, cruise, honkT; int col, line; };
static std::vector<Car> g_cars;
static const float LANE_W = -2.8f, LANE_E = 2.6f;  // westbound (-X) and eastbound (+X) lane centres
static const float CAR_WRAP = 400.0f;
static const char* HONK_LINES[] = {"HONK HONK!", "BEEP BEEP!", "Move it, kid!", "Get outta the street!", "Ya crazy?!"};
static void initCars() {
    g_cars.clear();
    for (int i = 0; i < 12; ++i) {
        Car c;
        bool west = i < 6;
        c.dir = west ? -1.0f : 1.0f;
        c.z = west ? LANE_W : LANE_E;
        c.x = -CAR_WRAP + (i % 6) * (2 * CAR_WRAP / 6) + frange(0, 50);
        c.cruise = frange(9.0f, 13.0f);
        c.speed = c.cruise;
        c.col = (i % 2 == 0) ? 0 : irange(1, 7);  // half the traffic is yellow cabs
        c.honkT = 0; c.line = 0;
        g_cars.push_back(c);
    }
}
static void updateCars(float dt, const V3& pp, bool playerInStreet) {
    for (Car& c : g_cars) {
        float target = c.cruise;
        for (const Car& o : g_cars) {
            if (&o == &c || o.z != c.z) continue;
            float gap = (o.x - c.x) * c.dir;
            if (gap > 0 && gap < 14.0f) target = std::min(target, o.speed * clampf((gap - 6.5f) / 6.0f, 0, 1));
        }
        float pg = (pp.x - c.x) * c.dir;
        if (playerInStreet && fabsf(pp.z - c.z) < 1.8f && pg > 0 && pg < 26.0f) {
            target = std::min(target, std::max(0.0f, (pg - 4.0f) * 0.6f));
            if (c.honkT <= 0 && pg < 20.0f) { c.honkT = 2.5f; c.line = irange(0, 4); }
        }
        float acc = target < c.speed ? 6.5f : 2.5f;
        c.speed = approachf(c.speed, target, acc * dt);
        c.x += c.dir * c.speed * dt;
        if (c.x > CAR_WRAP) c.x -= 2 * CAR_WRAP;
        if (c.x < -CAR_WRAP) c.x += 2 * CAR_WRAP;
        c.honkT = std::max(0.0f, c.honkT - dt);
    }
}
static void drawCars() {
    for (const Car& c : g_cars) {
        glPushMatrix();
        glTranslatef(c.x, 0.01f * sinf(g_time * 9.0f + c.x), c.z);
        if (c.dir < 0) glRotatef(180, 0, 1, 0);
        glCallList(g_carLists[c.col]);
        glPopMatrix();
    }
}

// ---- Figures (pedestrians and the skater share the rig) --------------------------
struct Outfit { Col shirt, pants, skin, hair, shoes, hat; int hatType; bool longSleeve; float baggy; };
enum { HAT_NONE, HAT_BEANIE, HAT_CAP, HAT_CAP_BACK };
struct Pose {
    V3 hipL, hipR, kneeL, kneeR, footL, footR, pelvis, chest, head, shL, shR, elL, elR, hdL, hdR;
    V3 fwd, right, footDirL, footDirR;
    float scale;
};
struct Frame {
    V3 o, r, u, f;
    float s;
    V3 at(float x, float y, float z) const { return o + (r * x + u * y + f * z) * s; }
};
static Frame makeFrame(const V3& o, float yaw, float s) {
    Frame F;
    F.o = o; F.f = fwdFromYaw(yaw); F.r = V3(-F.f.z, 0, F.f.x); F.u = V3(0, 1, 0); F.s = s;
    return F;
}
static Pose walkPose(const Frame& F, float phase, float amp) {
    Pose P;
    float sn = sinf(phase), cs = cosf(phase), y = amp > 0.01f ? fabsf(cs) * 0.035f : 0.0f;
    V3 fL(-0.11f, 0.05f + std::max(0.0f, cs) * 0.22f * amp, sn * amp), fR(0.11f, 0.05f + std::max(0.0f, -cs) * 0.22f * amp, -sn * amp);
    P.hipL = F.at(-0.1f, 0.9f + y, 0); P.hipR = F.at(0.1f, 0.9f + y, 0);
    P.footL = F.at(fL.x, fL.y, fL.z); P.footR = F.at(fR.x, fR.y, fR.z);
    P.kneeL = F.at(-0.105f, 0.47f + y * 0.5f + fL.y * 0.5f, fL.z * 0.5f + 0.06f);
    P.kneeR = F.at(0.105f, 0.47f + y * 0.5f + fR.y * 0.5f, fR.z * 0.5f + 0.06f);
    P.pelvis = F.at(0, 0.92f + y, 0);
    P.chest = F.at(0, 1.36f + y, 0.02f);
    P.head = F.at(0, 1.62f + y, 0.03f);
    P.shL = F.at(-0.2f, 1.4f + y, 0); P.shR = F.at(0.2f, 1.4f + y, 0);
    P.elL = F.at(-0.23f, 1.13f + y, -sn * amp * 0.45f); P.elR = F.at(0.23f, 1.13f + y, sn * amp * 0.45f);
    P.hdL = F.at(-0.24f, 0.88f + y, -sn * amp * 0.8f + 0.06f); P.hdR = F.at(0.24f, 0.88f + y, sn * amp * 0.8f + 0.06f);
    P.fwd = F.f; P.right = F.r; P.footDirL = P.footDirR = F.f; P.scale = F.s;
    return P;
}
static void drawEllipsoidYaw(const V3& c, float yaw, float rx, float ry, float rz) {
    glPushMatrix();
    glTranslatef(c.x, c.y, c.z);
    glRotatef(degf(yaw), 0, 1, 0);
    glScalef(rx, ry, rz);
    glCallList(g_sphereList);
    glPopMatrix();
}
static void drawFigure(const Pose& P, const Outfit& o) {
    const float s = P.scale, bg = o.baggy;
    texOff();
    glc(o.pants);
    for (int side = 0; side < 2; ++side) {
        const V3& h = side ? P.hipR : P.hipL;
        const V3& k = side ? P.kneeR : P.kneeL;
        const V3& f = side ? P.footR : P.footL;
        drawCyl(h, k, 0.085f * s * bg, 0.075f * s * bg, 8, false);
        drawSphere(k, 0.075f * s * bg);
        drawCyl(k, f + V3(0, 0.08f * s, 0), 0.075f * s * bg, 0.08f * s * bg, 8, true);
    }
    drawCyl(P.hipL, P.hipR, 0.105f * s * bg, 0.105f * s * bg, 8, true);
    glc(o.shoes);
    for (int side = 0; side < 2; ++side) {
        const V3& f = side ? P.footR : P.footL;
        const V3& fd = side ? P.footDirR : P.footDirL;
        drawEllipsoidYaw(f + fd * (0.04f * s), yawFromDir(fd.x, fd.z), 0.065f * s, 0.06f * s, 0.15f * s);
    }
    glc(o.shirt);
    drawCyl(P.pelvis, P.chest, 0.15f * s * bg, 0.17f * s * bg, 10, true);
    drawCyl(P.shL, P.shR, 0.075f * s * bg, 0.075f * s * bg, 8, false);
    drawSphere(P.shL, 0.08f * s * bg);
    drawSphere(P.shR, 0.08f * s * bg);
    for (int side = 0; side < 2; ++side) {
        const V3& sh = side ? P.shR : P.shL;
        const V3& el = side ? P.elR : P.elL;
        const V3& hd = side ? P.hdR : P.hdL;
        glc(o.shirt);
        drawCyl(sh, el, 0.06f * s * bg, 0.055f * s * bg, 7, false);
        drawSphere(el, 0.052f * s * bg);
        glc(o.longSleeve ? o.shirt : o.skin);
        drawCyl(el, hd, 0.048f * s, 0.042f * s, 7, false);
        glc(o.skin);
        drawSphere(hd, 0.05f * s);
    }
    glc(o.skin);
    drawCyl(P.chest, P.head, 0.05f * s, 0.05f * s, 6, false);
    drawSphere(P.head, 0.11f * s);
    const V3 up(0, 1, 0);
    float yaw = yawFromDir(P.fwd.x, P.fwd.z);
    glc(Col(0.05f, 0.05f, 0.05f));
    drawSphere(P.head + P.fwd * (0.098f * s) + P.right * (0.038f * s) + up * (0.02f * s), 0.014f * s);
    drawSphere(P.head + P.fwd * (0.098f * s) - P.right * (0.038f * s) + up * (0.02f * s), 0.014f * s);
    glc(o.hair);
    drawSphere(P.head - P.fwd * (0.02f * s) + up * (0.02f * s), 0.112f * s);
    if (o.hatType != HAT_NONE) {
        glc(o.hat);
        if (o.hatType == HAT_BEANIE) {
            drawEllipsoidYaw(P.head + up * (0.05f * s), yaw, 0.118f * s, 0.1f * s, 0.118f * s);
        } else {
            drawEllipsoidYaw(P.head + up * (0.05f * s), yaw, 0.116f * s, 0.075f * s, 0.116f * s);
            V3 bd = o.hatType == HAT_CAP ? P.fwd : -P.fwd;
            drawEllipsoidYaw(P.head + bd * (0.12f * s) + up * (0.045f * s), yaw, 0.08f * s, 0.012f * s, 0.085f * s);
        }
    }
}
// Ground shadow of a figure: each limb projected along the sun onto the ground.
static V3 projShadow(const V3& p, float fixedH, bool field) {
    float h = fixedH;
    if (field) {
        h = queryGround(p.x, p.z, p.y + 0.05f).h;
        for (int i = 0; i < 2; ++i) {
            float t = (p.y - h) / SUN_DIR.y;
            h = queryGround(p.x - SUN_DIR.x * t, p.z - SUN_DIR.z * t, p.y + 0.05f).h;
        }
    }
    float t = (p.y - h) / SUN_DIR.y;
    return V3(p.x - SUN_DIR.x * t, h + 0.015f, p.z - SUN_DIR.z * t);
}
static void shadowSeg(const V3& a, const V3& b, float w, float fixedH, bool field) {
    V3 A = projShadow(a, fixedH, field), B = projShadow(b, fixedH, field);
    float dx = B.x - A.x, dz = B.z - A.z, l = len2D(dx, dz), nx = w, nz = 0;
    if (l > 1e-4f) { nx = -dz / l * w; nz = dx / l * w; }
    glVertex3f(A.x + nx, A.y, A.z + nz); glVertex3f(A.x - nx, A.y, A.z - nz);
    glVertex3f(B.x - nx, B.y, B.z - nz); glVertex3f(B.x + nx, B.y, B.z + nz);
}
static void figureShadow(const Pose& P, float fixedH, bool field) {
    float s = P.scale;
    glBegin(GL_QUADS);
    shadowSeg(P.hipL, P.kneeL, 0.08f * s, fixedH, field); shadowSeg(P.kneeL, P.footL, 0.075f * s, fixedH, field);
    shadowSeg(P.hipR, P.kneeR, 0.08f * s, fixedH, field); shadowSeg(P.kneeR, P.footR, 0.075f * s, fixedH, field);
    shadowSeg(P.pelvis, P.chest, 0.16f * s, fixedH, field);
    shadowSeg(P.shL, P.shR, 0.07f * s, fixedH, field);
    shadowSeg(P.shL, P.elL, 0.05f * s, fixedH, field); shadowSeg(P.elL, P.hdL, 0.045f * s, fixedH, field);
    shadowSeg(P.shR, P.elR, 0.05f * s, fixedH, field); shadowSeg(P.elR, P.hdR, 0.045f * s, fixedH, field);
    shadowSeg(P.chest, P.head + V3(0, 0.1f * s, 0), 0.1f * s, fixedH, field);
    glEnd();
}

// ---- Pedestrians -----------------------------------------------------------------
enum NpcKind { NK_LANE, NK_LOOP, NK_IDLE };
enum IdleStyle { IS_STAND, IS_VENDOR, IS_PHONE };
struct Npc {
    int kind, idle;
    float along, laneZ, x0, x1, dir;  // lane walkers: position along X, lane centre, bounds, direction
    float speed, phase, yaw;
    float side, sideTarget, sideHold;  // sidestep to dodge the skater
    float wait, stun, cheerT, bubbleT;
    const char* bubble;
    float px, pz, tx, tz;  // resolved position and travel direction
    float faceYaw;         // smoothed facing used for rendering
    bool walking;
    Outfit look;
    float scale;
};
static std::vector<Npc> g_npcs;
static const char* BUMP_LINES[] = {"Hey! I'm walkin' here!", "Watch it, kid!", "Get outta here!", "You almost hit me!",
                                   "Not on the sidewalk!", "Fuggedaboutit!", "Are you serious?!", "Ay! Watch where you're goin'!",
                                   "I'm callin' the cops!", "This is a sidewalk, genius!"};
static const char* CHEER_LINES[] = {"Yo, that was sick!", "Ayyy!", "Do it again!", "That's what I'm talkin' about!",
                                    "Nice one, kid!", "Tight!", "Ha! That's ill!", "Somebody get that on tape!"};
static const float LOOP_X0 = -18, LOOP_X1 = 22, LOOP_Z0 = 19, LOOP_Z1 = 33;  // clear of the Banks landing zone
static void loopPos(float s, float& x, float& z, float& tx, float& tz) {
    const float W = LOOP_X1 - LOOP_X0, H = LOOP_Z1 - LOOP_Z0, P = 2 * (W + H);
    s = fmodf(s, P);
    if (s < 0) s += P;
    if (s < W) { x = LOOP_X0 + s; z = LOOP_Z0; tx = 1; tz = 0; return; }
    s -= W;
    if (s < H) { x = LOOP_X1; z = LOOP_Z0 + s; tx = 0; tz = 1; return; }
    s -= H;
    if (s < W) { x = LOOP_X1 - s; z = LOOP_Z1; tx = -1; tz = 0; return; }
    s -= W;
    x = LOOP_X0; z = LOOP_Z1 - s; tx = 0; tz = -1;
}
static Outfit makeOutfit(int i) {
    static const Col shirts[10] = {Col(0.85f, 0.15f, 0.15f), Col(0.15f, 0.3f, 0.7f), Col(0.95f, 0.95f, 0.92f), Col(0.1f, 0.1f, 0.1f),
                                   Col(0.95f, 0.75f, 0.15f), Col(0.3f, 0.55f, 0.3f), Col(0.55f, 0.3f, 0.6f), Col(0.5f, 0.5f, 0.55f),
                                   Col(0.95f, 0.5f, 0.15f), Col(0.2f, 0.6f, 0.7f)};
    static const Col pants[5] = {Col(0.2f, 0.3f, 0.5f), Col(0.12f, 0.12f, 0.14f), Col(0.6f, 0.52f, 0.38f), Col(0.35f, 0.35f, 0.38f), Col(0.15f, 0.2f, 0.35f)};
    static const Col skins[5] = {Col(0.96f, 0.8f, 0.66f), Col(0.85f, 0.65f, 0.48f), Col(0.6f, 0.42f, 0.3f), Col(0.42f, 0.28f, 0.2f), Col(0.9f, 0.72f, 0.56f)};
    static const Col hairs[5] = {Col(0.08f, 0.06f, 0.05f), Col(0.3f, 0.2f, 0.1f), Col(0.6f, 0.45f, 0.2f), Col(0.5f, 0.5f, 0.5f), Col(0.15f, 0.1f, 0.08f)};
    Outfit o;
    o.shirt = shirts[(int)(hash01(i, 1, 77) * 10)];
    o.pants = pants[(int)(hash01(i, 2, 77) * 5)];
    o.skin = skins[(int)(hash01(i, 3, 77) * 5)];
    o.hair = hairs[(int)(hash01(i, 4, 77) * 5)];
    o.shoes = hash01(i, 5, 77) < 0.5f ? Col(0.92f, 0.92f, 0.9f) : Col(0.12f, 0.1f, 0.1f);
    o.hat = shirts[(int)(hash01(i, 6, 77) * 10)];
    int h = (int)(hash01(i, 7, 77) * 6);
    o.hatType = h < 3 ? HAT_NONE : (h == 3 ? HAT_BEANIE : (h == 4 ? HAT_CAP : HAT_CAP_BACK));
    o.longSleeve = hash01(i, 8, 77) < 0.5f;
    o.baggy = 1.0f + 0.3f * hash01(i, 9, 77);
    return o;
}
static Npc baseNpc(int kind, int seed) {
    Npc n{};
    n.kind = kind; n.idle = IS_STAND; n.dir = 1; n.bubble = "";
    n.speed = frange(1.1f, 1.6f);
    n.phase = frand() * 6.0f;
    n.look = makeOutfit(seed);
    n.scale = frange(0.93f, 1.08f);
    n.tx = 1;
    return n;
}
static void initNpcs() {
    g_npcs.clear();
    int seed = 1;
    auto lane = [&](float z, float startX, float dir) {
        Npc n = baseNpc(NK_LANE, seed++);
        n.laneZ = z; n.x0 = -53.0f; n.x1 = 53.0f; n.along = startX; n.dir = dir;
        g_npcs.push_back(n);
    };
    lane(-10.4f, -30, 1); lane(-10.4f, 5, -1); lane(-10.4f, 38, 1); lane(-12.3f, -12, -1); lane(-12.3f, 22, 1);
    lane(12.0f, -40, 1); lane(12.0f, 15, -1); lane(13.2f, -5, 1); lane(13.2f, 35, -1);
    lane(41.0f, -35, 1); lane(41.0f, 20, -1); lane(41.9f, 0, 1);
    for (int i = 0; i < 5; ++i) {
        Npc n = baseNpc(NK_LOOP, seed++);
        n.along = i * 23.0f; n.dir = (i % 2) ? -1.0f : 1.0f;
        g_npcs.push_back(n);
    }
    auto idle = [&](float x, float z, float yaw, int style) {
        Npc n = baseNpc(NK_IDLE, seed++);
        n.px = x; n.pz = z; n.yaw = yaw; n.idle = style;
        n.tx = sinf(yaw); n.tz = cosf(yaw); n.faceYaw = yaw;
        g_npcs.push_back(n);
    };
    idle(-5.0f, 40.15f, PI, IS_VENDOR);
    idle(-42.5f, -13.3f, 0.0f, IS_STAND);
    idle(7.6f, 9.55f, PI, IS_PHONE);
    idle(24.0f, 30.0f, PI * 0.5f, IS_STAND);
    idle(-24.0f, 34.5f, PI * 0.75f, IS_STAND);
}
static void npcSay(Npc& n, const char* line, float t) { n.bubble = line; n.bubbleT = t; }
static void npcCheer(const V3& at, float radius) {
    for (Npc& n : g_npcs) {
        if (len2D(n.px - at.x, n.pz - at.z) > radius || n.stun > 0) continue;
        if (frand() < 0.6f) { n.cheerT = 1.8f; npcSay(n, CHEER_LINES[irange(0, 7)], 2.2f); }
    }
}
static void updateNpcs(float dt, const V3& pp, const V3& pv) {
    float pspeed = len2D(pv.x, pv.z);
    for (Npc& n : g_npcs) {
        n.bubbleT = std::max(0.0f, n.bubbleT - dt);
        n.cheerT = std::max(0.0f, n.cheerT - dt);
        bool moving = false;
        if (n.stun > 0) n.stun -= dt;
        else if (n.wait > 0) n.wait -= dt;
        else if (n.kind == NK_LANE) {
            n.along += n.dir * n.speed * dt;
            if (n.along > n.x1) { n.along = n.x1; n.dir = -1; n.wait = frange(0.5f, 2.0f); }
            if (n.along < n.x0) { n.along = n.x0; n.dir = 1; n.wait = frange(0.5f, 2.0f); }
            moving = true;
        } else if (n.kind == NK_LOOP) {
            n.along += n.dir * n.speed * dt;
            moving = true;
        }
        float bx = n.px, bz = n.pz, tx = n.tx, tz = n.tz;
        if (n.kind == NK_LANE) { bx = n.along; bz = n.laneZ; tx = n.dir; tz = 0; }
        else if (n.kind == NK_LOOP) { loopPos(n.along, bx, bz, tx, tz); tx *= n.dir; tz *= n.dir; }
        if (n.kind != NK_IDLE) {
            float rx = pp.x - n.px, rz = pp.z - n.pz, d = len2D(rx, rz);
            float sx = -tz, sz = tx;  // right-hand side of travel
            bool threat = d < 5.0f && pspeed > 2.5f && (pv.x * -rx + pv.z * -rz) > 0;
            if (threat) { n.sideTarget = (rx * sx + rz * sz > 0) ? -0.9f : 0.9f; n.sideHold = 1.5f; }
            else if ((n.sideHold -= dt) <= 0) n.sideTarget = 0;
            n.side = approachf(n.side, n.sideTarget, 2.0f * dt);
            n.px = bx + sx * n.side;
            n.pz = bz + sz * n.side;
            n.tx = tx; n.tz = tz;
        }
        float targetYaw = n.kind == NK_IDLE ? n.yaw : yawFromDir(n.tx, n.tz);
        if (n.stun > 0 || n.cheerT > 0) targetYaw = yawFromDir(pp.x - n.px, pp.z - n.pz);
        n.faceYaw = wrapPi(n.faceYaw + wrapPi(targetYaw - n.faceYaw) * expK(8.0f, dt));
        n.walking = moving;
        if (moving) n.phase += 2 * PI * n.speed / (4 * 0.32f) * dt;
    }
}

static Pose npcPose(const Npc& n) {
    Frame F = makeFrame(V3(n.px, CURB_H, n.pz), n.faceYaw, n.scale);
    if (n.walking && n.stun <= 0) return walkPose(F, n.phase, 0.3f);
    Pose P = walkPose(F, 0, 0);
    float br = 0.01f * sinf(g_time * 2.0f + n.scale * 20.0f);
    P.chest += V3(0, br, 0); P.head += V3(0, br, 0);
    if (n.stun > 0) {  // arms thrown up in protest
        P.elL = F.at(-0.36f, 1.5f, 0.1f); P.elR = F.at(0.36f, 1.5f, 0.1f);
        P.hdL = F.at(-0.3f, 1.78f, 0.15f); P.hdR = F.at(0.3f, 1.78f, 0.15f);
    } else if (n.cheerT > 0) {
        float b = fabsf(sinf(g_time * 9.0f)) * 0.06f;
        P.elR = F.at(0.3f, 1.62f + b, 0.05f); P.hdR = F.at(0.28f, 1.92f + b, 0.08f);
        P.elL = F.at(-0.3f, 1.62f + b, 0.05f); P.hdL = F.at(-0.26f, 1.9f + b, 0.08f);
    } else if (n.idle == IS_VENDOR) {
        P.elL = F.at(-0.25f, 1.1f, 0.15f); P.hdL = F.at(-0.2f, 1.08f, 0.38f);
        P.elR = F.at(0.25f, 1.1f, 0.15f); P.hdR = F.at(0.2f, 1.08f, 0.38f);
    } else if (n.idle == IS_PHONE) {
        P.elR = F.at(0.3f, 1.3f, 0.05f); P.hdR = F.at(0.12f, 1.6f, 0.06f);
    }
    return P;
}
static void drawNpcs() {
    for (const Npc& n : g_npcs) drawFigure(npcPose(n), n.look);
}
static void drawNpcShadows() {
    for (const Npc& n : g_npcs) figureShadow(npcPose(n), CURB_H, false);
}
static void drawCarShadows(const V3& cam) {
    std::vector<V3> pts;
    for (const Car& c : g_cars) {
        if (fabsf(c.x - cam.x) > 160.0f) continue;
        float cab = c.dir > 0 ? -0.07f : 0.07f;
        pts.clear();
        boxPoints(pts, c.x - 2.3f, 0.0f, c.z - 0.9f, c.x + 2.3f, 0.95f, c.z + 0.9f);
        boxPoints(pts, c.x + cab - 1.1f, 0.95f, c.z - 0.8f, c.x + cab + 1.05f, 1.45f, c.z + 0.8f);
        fillPoly(projectHull(pts, 0.0f), 0.014f);
    }
}

// ---- Pigeons -----------------------------------------------------------------------
enum { PG_GROUND, PG_FLY, PG_RETURN };
struct Pigeon { V3 p, v, home; float yaw, timer, flap, peck; int state; };
static std::vector<Pigeon> g_pigeons;
static void initPigeons() {
    g_pigeons.clear();
    const V3 spots[4] = {V3(-6, CURB_H, 17.5f), V3(-30, CURB_H, -11.2f), V3(15, CURB_H, 40.0f), V3(-9, CURB_H, 33.5f)};
    for (const V3& s : spots)
        for (int i = 0; i < 6; ++i) {
            Pigeon g{};
            g.home = s + V3(frange(-1.6f, 1.6f), 0, frange(-0.9f, 0.9f));
            g.p = g.home;
            g.yaw = frand() * 2 * PI;
            g.timer = frand() * 2;
            g.state = PG_GROUND;
            g_pigeons.push_back(g);
        }
}
static void updatePigeons(float dt, const V3& pp, float pspeed) {
    for (Pigeon& g : g_pigeons) {
        g.flap += dt * (g.state == PG_GROUND ? 0.0f : 22.0f);
        float d = len2D(pp.x - g.p.x, pp.z - g.p.z);
        if (g.state == PG_GROUND) {
            g.timer -= dt;
            g.peck = std::max(0.0f, g.peck - dt);
            if (g.timer <= 0) {
                g.timer = frange(0.6f, 2.0f);
                if (frand() < 0.45f) g.peck = 0.5f;
                else {
                    V3 to = g.home - g.p;
                    float a = len2D(to.x, to.z) > 1.2f ? yawFromDir(to.x, to.z) + frange(-0.5f, 0.5f) : frand() * 2 * PI;
                    g.yaw = a;
                    g.v = fwdFromYaw(a) * frange(0.25f, 0.45f);
                }
            }
            if (g.peck > 0) g.v *= 0.0f;
            g.p += g.v * dt;
            g.v *= 1.0f / (1.0f + 1.5f * dt);
            if (d < 2.8f + pspeed * 0.25f && (pspeed > 1.0f || d < 1.2f)) {
                g.state = PG_FLY;
                g.timer = frange(3.5f, 6.0f);
                V3 away = normalize(V3(g.p.x - pp.x, 0, g.p.z - pp.z));
                float a = yawFromDir(away.x, away.z) + frange(-0.8f, 0.8f);
                g.v = fwdFromYaw(a) * frange(3.5f, 5.0f) + V3(0, frange(3.0f, 4.5f), 0);
                g.yaw = a;
            }
        } else if (g.state == PG_FLY) {
            g.timer -= dt;
            g.v.y = g.p.y < 9.0f ? lerpf(g.v.y, 2.2f, expK(1.5f, dt)) : lerpf(g.v.y, 0.0f, expK(2.0f, dt));
            float turn = 0.6f * dt, c = cosf(turn), s = sinf(turn);
            g.v = V3(g.v.x * c + g.v.z * s, g.v.y, -g.v.x * s + g.v.z * c);
            g.p += g.v * dt;
            g.yaw = yawFromDir(g.v.x, g.v.z);
            if (g.timer <= 0) g.state = PG_RETURN;
        } else {
            V3 to = g.home - g.p;
            float dh = len2D(to.x, to.z);
            if (dh < 0.3f && fabsf(to.y) < 0.4f) { g.p = g.home; g.v = V3(); g.state = PG_GROUND; g.timer = 1; continue; }
            V3 want = normalize(to) * std::min(5.0f, 1.0f + dh);
            g.v = lerpv(g.v, want, expK(2.0f, dt));
            g.p += g.v * dt;
            g.yaw = yawFromDir(g.v.x, g.v.z);
        }
        g.p.z = std::max(g.p.z, FACADE_Z + 0.4f);
        g.p.y = std::max(g.p.y, CURB_H);
    }
}
static void drawPigeons() {
    texOff();
    for (const Pigeon& g : g_pigeons) {
        bool air = g.state != PG_GROUND;
        glPushMatrix();
        glTranslatef(g.p.x, g.p.y, g.p.z);
        glRotatef(degf(g.yaw), 0, 1, 0);
        glc(Col(0.48f, 0.5f, 0.56f));
        drawSphere(V3(0, 0.11f, 0), 0.075f, 0.07f, 0.13f);
        drawCyl(V3(0, 0.12f, -0.1f), V3(0, 0.1f, -0.22f), 0.04f, 0.015f, 5, false);
        float pk = g.peck > 0 ? sinf(g.peck * 2 * PI) * 0.07f : 0.0f;
        glc(Col(0.3f, 0.42f, 0.38f));
        drawSphere(V3(0, 0.18f - pk, 0.1f + pk * 0.5f), 0.045f);
        glc(Col(0.35f, 0.36f, 0.4f));
        drawSphere(V3(0, 0.22f - pk * 1.6f, 0.15f + pk), 0.035f);
        glc(Col(0.85f, 0.6f, 0.4f));
        drawCyl(V3(0, 0.215f - pk * 1.6f, 0.18f + pk), V3(0, 0.21f - pk * 1.6f, 0.22f + pk), 0.008f, 0.002f, 4, false);
        if (!air) {
            drawCyl(V3(-0.03f, 0.06f, 0), V3(-0.03f, 0, 0.01f), 0.006f, 0.006f, 3, false);
            drawCyl(V3(0.03f, 0.06f, 0), V3(0.03f, 0, 0.01f), 0.006f, 0.006f, 3, false);
        }
        glc(Col(0.42f, 0.44f, 0.5f));
        float ang = air ? sinf(g.flap) * 0.9f : -0.15f, span = air ? 0.24f : 0.1f;
        glBegin(GL_TRIANGLES);
        for (int s = -1; s <= 1; s += 2) {
            V3 root(s * 0.05f, 0.14f, 0.05f), root2(s * 0.05f, 0.14f, -0.08f);
            V3 tip(s * (0.05f + span * cosf(ang)), 0.14f + span * sinf(ang), -0.02f);
            nrm(V3(0, 1, 0));
            vtx(root); vtx(tip); vtx(root2);
        }
        glEnd();
        glPopMatrix();
    }
}

static void updateWorld(float dt, const V3& pp, const V3& pv, bool playerInStreet) {
    g_time += dt;
    updateEmitters(dt);
    updateParticles(dt);
    updateCars(dt, pp, playerInStreet);
    updateNpcs(dt, pp, pv);
    updatePigeons(dt, pp, len2D(pv.x, pv.z));
    g_ferryX += 3.0f * dt;
    if (g_ferryX > 420.0f) g_ferryX = -420.0f;
}

// ----------------------------------------------------------------------------
//  Player state
// ----------------------------------------------------------------------------
enum PlayerState { PS_GROUND, PS_AIR, PS_GRIND, PS_BAIL };
enum FlipKind { FK_NONE = -1, FK_KICK, FK_HEEL, FK_SHOVE, FK_TRE, FK_IMPOSS };
struct FlipDef { const char* name; float pts, dur; };
static const FlipDef FLIPS[5] = {{"Kickflip", 300, 0.40f}, {"Heelflip", 300, 0.40f}, {"Pop Shove-It", 200, 0.34f},
                                 {"360 Flip", 550, 0.50f}, {"Impossible", 500, 0.48f}};
static const float FLIP_MULT[5] = {0.0f, 1.0f, 2.6f, 4.5f, 7.0f};  // single, double, triple, quad
enum GrindKind { GK_5050, GK_50, GK_NOSE, GK_BOARD, GK_NOSESLIDE, GK_TAILSLIDE, GK_CROOK, GK_SMITH, GK_FEEBLE, GK_COUNT };
struct GrindDef { const char* name; float base, rate, pitch, yaw, roll; bool slide; };
static const GrindDef GRINDS[GK_COUNT] = {
    {"50-50", 100, 160, 0.0f, 0.0f, 0.0f, false},          {"5-0", 150, 200, 0.2f, 0.0f, 0.0f, false},
    {"Nosegrind", 150, 200, -0.2f, 0.0f, 0.0f, false},     {"Boardslide", 150, 200, 0.0f, 0.0f, 0.0f, true},
    {"Noseslide", 200, 240, -0.12f, 0.0f, 0.0f, true},     {"Tailslide", 200, 240, 0.12f, 0.0f, 0.0f, true},
    {"Crooked Grind", 250, 260, -0.15f, 0.35f, 0.1f, false}, {"Smith Grind", 250, 260, 0.14f, 0.0f, 0.3f, false},
    {"Feeble Grind", 250, 260, 0.12f, -0.25f, -0.25f, false}};

struct Player {
    int state;
    V3 pos, vel;
    float speed, yaw;  // ground speed (>= 0) and travel heading
    bool fakie;
    float spin, spinVel;  // air rotation of the body relative to travel
    float crouch, charge;
    bool charging;
    float pushT, squash, popT;
    float boardPitch, boardRoll, visYaw, flipRoll, flipYaw, flipPitch, boardLift;
    int flipKind, flipCount, flipEntry;
    float flipT, flipDur;
    int grabKind, grabEntry;
    int airEntry;  // combo entry the air's spin is credited to
    float airTime;
    int rail, lastRail, grindKind, grindEntry, slideSign;
    float railT, railDir, railSpeed, railCool, anyRailCool, grindTime, grindDist, sparkAcc;
    float bal, balVel;
    bool manual, noseManual;
    int manualEntry;
    float manualTime;
    int groundSolid;
    float bankTimer;
    bool bankPending;
    float splashAcc;
    V3 takeoff;
    int takeoffSolid;
    bool fromRail, fromManual, fakieAtTakeoff, overFountain, overPit, overCar, overTaxi;
    float bailT;
    bool bailSafe;
    V3 boardPos, boardVel;
    float boardSpin;
    V3 safePos;
    float safeYaw, safeTimer;
};
static Player P;

// ----------------------------------------------------------------------------
//  Scoring: combo list, popups, gaps
// ----------------------------------------------------------------------------
struct ComboEntry {
    std::string name;
    float base, rate, time, mult;  // points = (base + rate * time) * mult
    bool live;                     // still accruing time (grinds, manuals, grabs)
};
static std::vector<ComboEntry> g_combo;
static long g_score = 0, g_bestCombo = 0;

// Repeating a trick inside one combo is worth less each time.
static float repeatMult(const std::string& name, int skip) {
    int n = 0;
    for (int i = 0; i < (int)g_combo.size(); ++i)
        if (i != skip && g_combo[i].name == name) ++n;
    return std::max(0.1f, powf(0.6f, (float)n));
}
static int comboAdd(const std::string& name, float base, float rate, bool live) {
    ComboEntry c;
    c.name = name; c.base = base; c.rate = rate; c.time = 0; c.live = live;
    c.mult = repeatMult(name, -1);
    g_combo.push_back(c);
    return (int)g_combo.size() - 1;
}
static void comboSet(int i, const std::string& name, float base, float rate) {
    if (i < 0 || i >= (int)g_combo.size()) return;
    ComboEntry& c = g_combo[i];
    c.name = name; c.base = base; c.rate = rate;
    c.mult = repeatMult(name, i);
}
static void comboFreeze(int i) { if (i >= 0 && i < (int)g_combo.size()) g_combo[i].live = false; }
static float entryPts(const ComboEntry& c) { return (c.base + c.rate * c.time) * c.mult; }
static float comboBase() { float s = 0; for (const ComboEntry& c : g_combo) s += entryPts(c); return s; }
static int comboMult() { return (int)g_combo.size(); }
static void comboTick(float dt) { for (ComboEntry& c : g_combo) if (c.live) c.time += dt; }

struct Popup { std::string big, small; Col col; float t, dur; };
static Popup g_popup = {"", "", Col(), 0, 0};
struct Toast { std::string text; Col col; float t; };
static std::vector<Toast> g_toasts;
static void toast(const std::string& s, const Col& c) {
    g_toasts.push_back({s, c, 2.4f});
    if (g_toasts.size() > 5) g_toasts.erase(g_toasts.begin());
}
static void popup(const std::string& big, const std::string& small, const Col& c, float dur) { g_popup = {big, small, c, dur, dur}; }
static std::string commas(long v) {
    std::string s = std::to_string(v < 0 ? -v : v), out;
    int n = 0;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        out.insert(out.begin(), s[i]);
        if (++n % 3 == 0 && i > 0) out.insert(out.begin(), ',');
    }
    return v < 0 ? "-" + out : out;
}
// Trick names joined with " + ", keeping the most recent ones when too long.
static std::string comboText(size_t maxLen) {
    std::string s;
    for (int i = (int)g_combo.size() - 1; i >= 0; --i) {
        std::string piece = g_combo[i].name + (s.empty() ? "" : " + ");
        if (s.size() + piece.size() > maxLen) { s = "... " + s; break; }
        s = piece + s;
    }
    return s;
}

enum GapId { G_STAIRS, G_DROP, G_FOUNTAIN, G_TAXI, G_CAR, G_SURF, G_HUDSON, G_TRANSFER, G_KICKBAR, G_HYDRANT, G_SUBWAY, G_KICKMAN, G_HANDRAIL, G_COUNT };
struct GapDef { const char* name; int pts; };
static const GapDef GAPS[G_COUNT] = {
    {"Banks Stair Set", 500}, {"Banks Drop", 250},     {"Fountain Gap", 600},   {"Taxi Hop", 500},
    {"Car Hop", 400},         {"Car Surfin'", 300},    {"Hudson Rail", 700},    {"Rail Transfer", 300},
    {"Kicker To Bar", 400},   {"Hydrant Splash", 250}, {"Subway Hop", 350},     {"Kicker To Manual", 300},
    {"Handrail Hero", 600}};
static bool g_gapFound[G_COUNT];
static bool g_gapThisAir[G_COUNT];  // each gap counts once per air / grind
static int gapsFound() { int n = 0; for (bool b : g_gapFound) n += b; return n; }
static void awardGap(int g) {
    if (g_gapThisAir[g]) return;
    g_gapThisAir[g] = true;
    comboAdd(std::string("Gap: ") + GAPS[g].name, (float)GAPS[g].pts, 0, false);
    if (!g_gapFound[g]) {
        g_gapFound[g] = true;
        toast(std::string("NEW GAP!  ") + GAPS[g].name + "   " + std::to_string(gapsFound()) + "/" + std::to_string((int)G_COUNT),
              Col(0.4f, 1.0f, 0.6f));
    } else {
        toast(std::string("GAP: ") + GAPS[g].name, Col(0.6f, 0.9f, 1.0f));
    }
}

static void comboBank() {
    if (g_combo.empty()) return;
    int n = comboMult();
    long pts = (long)(comboBase() * n + 0.5f);
    g_score += pts;
    const char* tier;
    Col col;
    if (n == 1 && pts < 600) { tier = ""; col = Col(1.0f, 1.0f, 1.0f); }
    else if (pts < 1500) { tier = "NICE!"; col = Col(0.6f, 1.0f, 0.6f); }
    else if (pts < 4000) { tier = "SWEET!"; col = Col(0.4f, 0.9f, 1.0f); }
    else if (pts < 10000) { tier = "SICK!!"; col = Col(1.0f, 0.85f, 0.2f); }
    else if (pts < 25000) { tier = "INSANE!!"; col = Col(1.0f, 0.55f, 0.15f); }
    else if (pts < 60000) { tier = "BIG APPLE BANGER!!!"; col = Col(1.0f, 0.35f, 0.65f); }
    else { tier = "NYC LEGEND!!!"; col = Col(0.8f, 0.55f, 1.0f); }
    std::string big = (tier[0] ? std::string(tier) + "  " : std::string()) + "+" + commas(pts);
    std::string small = comboText(70) + (n > 1 ? "   x" + std::to_string(n) : "");
    popup(big, small, col, n > 1 ? 2.8f : 1.6f);
    if (pts > g_bestCombo) {
        if (g_bestCombo > 0 && n > 2) toast("NEW BEST COMBO!", Col(1.0f, 0.85f, 0.2f));
        g_bestCombo = pts;
    }
    if (pts >= 3000) npcCheer(P.pos, 22.0f);
    g_combo.clear();
}
static void comboLose(const char* why) {
    long lost = (long)(comboBase() * std::max(1, comboMult()) + 0.5f);
    popup("BAIL!", std::string(why) + (lost > 0 ? "    combo lost: " + commas(lost) : ""), Col(1.0f, 0.25f, 0.2f), 2.4f);
    g_combo.clear();
}

// ----------------------------------------------------------------------------
//  Player helpers
// ----------------------------------------------------------------------------
static const V3 SPAWN_POS(14.0f, CURB_H, 17.5f);
static const float SPAWN_YAW = PI * 0.5f;  // facing +X, lined up with the kicker -> manual pad
static int g_kicker1 = -1, g_kicker2 = -1, g_manualPad = -1, g_platform = -1;
static int findSolidAt(float x0, float z0) {
    for (size_t i = 0; i < g_solids.size(); ++i)
        if (fabsf(g_solids[i].x0 - x0) < 0.01f && fabsf(g_solids[i].z0 - z0) < 0.01f) return (int)i;
    return -1;
}
static void initGameRefs() {
    g_kicker1 = findSolidAt(26.0f, 16.5f);
    g_kicker2 = findSolidAt(44.0f, 25.0f);
    g_manualPad = findSolidAt(33.0f, 16.8f);
    g_platform = findSolidAt(-46.0f, 18.0f);
}
static void resetPlayer(bool clearCombo) {
    P = Player{};
    P.state = PS_GROUND;
    P.pos = SPAWN_POS;
    P.yaw = SPAWN_YAW;
    P.flipKind = FK_NONE;
    P.flipEntry = P.grabEntry = P.airEntry = P.grindEntry = P.manualEntry = -1;
    P.grabKind = -1;
    P.rail = P.lastRail = -1;
    P.groundSolid = P.takeoffSolid = -1;
    P.safePos = SPAWN_POS;
    P.safeYaw = SPAWN_YAW;
    if (clearCombo) g_combo.clear();
}

static const float PUSH_ACC = 7.5f, PUSH_MAX = 10.0f, MAX_SPEED = 18.0f, ROLL_FRIC = 0.35f, DRAG = 0.006f;
static const float BRAKE = 9.0f, CHARGE_TIME = 0.45f, SPIN_MAX = 8.5f, SPIN_ACC = 22.0f, SNAP_DROP = 0.22f;
static const float POP_MIN = 5.2f, POP_MAX = 7.4f;
static const float SPIN_PTS[8] = {0, 150, 400, 800, 1300, 2000, 2800, 3800};

static std::vector<int> g_stepPresses;  // key presses delivered to this simulation step
static bool pressed(int k) { return std::find(g_stepPresses.begin(), g_stepPresses.end(), k) != g_stepPresses.end(); }

static void beginAir(const V3& vel, bool fromRail) {
    P.fromManual = P.manual;
    if (P.manual) { comboFreeze(P.manualEntry); P.manual = false; }
    P.state = PS_AIR;
    P.vel = vel;
    P.spin = 0; P.spinVel = 0;
    P.flipKind = FK_NONE; P.flipEntry = -1; P.flipCount = 0; P.flipT = 0;
    P.grabKind = -1; P.grabEntry = -1;
    P.airEntry = -1; P.airTime = 0;
    P.takeoff = P.pos;
    P.takeoffSolid = fromRail ? -1 : P.groundSolid;
    P.fromRail = fromRail;
    P.fakieAtTakeoff = P.fakie;
    P.overFountain = P.overPit = P.overCar = P.overTaxi = false;
    for (bool& b : g_gapThisAir) b = false;
}
static void doOllie() {
    float pop = lerpf(POP_MIN, POP_MAX, P.charge);
    V3 f = fwdFromYaw(P.yaw);
    GroundHit g = queryGround(P.pos.x, P.pos.z, P.pos.y + 0.05f);
    float slopeV = std::max(0.0f, (g.gx * f.x + g.gz * f.z) * P.speed);
    bool fromMan = P.manual;
    if (P.bankPending) comboBank();  // only manuals link tricks on the ground
    P.bankPending = false;
    beginAir(V3(f.x * P.speed, pop + slopeV, f.z * P.speed), false);
    P.popT = 0.18f;
    P.spinVel = ((kLeft() ? 1.0f : 0.0f) - (kRight() ? 1.0f : 0.0f)) * 3.0f;  // pre-wound spin
    if (!fromMan) P.airEntry = comboAdd("Ollie", 100, 0, false);
    emitDust(P.pos, 4, 0.6f);
}
// Credits `halves` x 180 degrees of rotation to the air's first trick (or as its own trick).
static void applySpin(int halves, float spinSign) {
    if (halves <= 0) return;
    bool fs = (spinSign > 0) != P.fakieAtTakeoff;
    std::string lbl = std::string(fs ? "FS " : "BS ") + std::to_string(halves * 180);
    float pts = SPIN_PTS[std::min(halves, 7)];
    if (P.airEntry >= 0 && P.airEntry < (int)g_combo.size()) {
        ComboEntry e = g_combo[P.airEntry];
        comboSet(P.airEntry, e.name == "Ollie" ? lbl : lbl + " " + e.name, e.base + pts, e.rate);
    } else {
        comboAdd(lbl, pts, 0, false);
    }
}
static std::string flipName(int kind, int count) {
    if (kind == FK_SHOVE) return count <= 1 ? std::string("Pop Shove-It") : std::to_string(180 * count) + " Shove-It";
    static const char* pre[5] = {"", "", "Double ", "Triple ", "Quad "};
    return std::string(pre[std::min(count, 4)]) + FLIPS[kind].name;
}
static void pressFlip(int kind) {
    if (P.grabKind >= 0) return;
    if (P.flipKind == kind && P.flipCount < 4 && P.flipT < P.flipDur * 0.8f) {  // tap again: double, triple...
        P.flipCount++;
        P.flipDur = FLIPS[kind].dur * (1.0f + 0.75f * (P.flipCount - 1));
        comboSet(P.flipEntry, flipName(kind, P.flipCount), FLIPS[kind].pts * FLIP_MULT[P.flipCount], 0);
        return;
    }
    if (P.flipKind != FK_NONE) return;  // finish the current flip first
    P.flipKind = kind; P.flipCount = 1; P.flipT = 0; P.flipDur = FLIPS[kind].dur;
    if (P.airEntry >= 0 && g_combo[P.airEntry].name == "Ollie") {
        comboSet(P.airEntry, FLIPS[kind].name, FLIPS[kind].pts, 0);
        P.flipEntry = P.airEntry;
    } else {
        P.flipEntry = comboAdd(FLIPS[kind].name, FLIPS[kind].pts, 0, false);
        if (P.airEntry < 0) P.airEntry = P.flipEntry;
    }
}
static void startGrab() {
    if (P.grabKind >= 0 || P.flipKind != FK_NONE) return;
    static const char* names[3] = {"Indy", "Nosegrab", "Tailgrab"};
    P.grabKind = kFwd() ? 1 : (kBack() ? 2 : 0);
    if (P.airEntry >= 0 && g_combo[P.airEntry].name == "Ollie") {
        comboSet(P.airEntry, names[P.grabKind], 200, 400);
        g_combo[P.airEntry].live = true;
        P.grabEntry = P.airEntry;
    } else {
        P.grabEntry = comboAdd(names[P.grabKind], 200, 400, true);
        if (P.airEntry < 0) P.airEntry = P.grabEntry;
    }
}
static void endGrab() {
    if (P.grabKind < 0) return;
    comboFreeze(P.grabEntry);
    P.grabKind = -1;
}

// ---- Grinds ------------------------------------------------------------------------
static void airGaps(const V3& at, bool onRail);
static void bail(const char* why, bool toSafe, const V3& kick);

static void startGrind(int ri, float t) {
    const Rail& r = g_rails[ri];
    V3 d = normalize(r.b - r.a);
    float dh = std::max(1e-4f, len2D(d.x, d.z));
    V3 dH(d.x / dh, 0, d.z / dh);
    float hs = len2D(P.vel.x, P.vel.z), along = P.vel.x * dH.x + P.vel.z * dH.z;
    P.railDir = along >= 0 ? 1.0f : -1.0f;
    float travelYaw = yawFromDir(dH.x * P.railDir, dH.z * P.railDir);
    // Board heading relative to the rail decides grind vs slide (sideways) vs fakie.
    float rel = wrapPi(P.yaw + P.spin - travelYaw);
    int q = (int)lroundf(rel / (PI * 0.5f));
    applySpin((int)lroundf(fabsf(P.spin) / (PI * 0.5f)) / 2, P.spin);
    bool slide = (q & 1) != 0;
    bool reversed = q == 2 || q == -2;
    if (reversed) P.fakie = !P.fakie;
    P.visYaw = reversed ? wrapPi(rel - PI) : rel;  // keep the board where it was; it eases into the grind pose
    P.slideSign = rel >= 0 ? 1 : -1;
    P.yaw = travelYaw;
    P.spin = 0; P.spinVel = 0;
    endGrab();
    int kind;
    if (slide) kind = kFwd() ? GK_NOSESLIDE : (kBack() ? GK_TAILSLIDE : GK_BOARD);
    else kind = kFwd() ? GK_NOSE : (kBack() ? GK_50 : GK_5050);
    if (P.fromRail && P.lastRail >= 0 && P.lastRail != ri && strcmp(g_rails[P.lastRail].name, r.name) != 0) awardGap(G_TRANSFER);
    if (P.takeoffSolid == g_kicker2 && g_kicker2 >= 0 && strcmp(r.name, "Flat Bar") == 0) awardGap(G_KICKBAR);
    airGaps(lerpv(r.a, r.b, t), true);
    P.state = PS_GRIND;
    P.rail = ri; P.railT = t;
    P.railSpeed = std::max(std::max(fabsf(along), 0.6f * hs), 3.0f);
    P.grindKind = kind;
    P.grindTime = 0; P.grindDist = 0; P.sparkAcc = 0;
    P.bal = frange(-0.15f, 0.15f); P.balVel = frange(-0.2f, 0.2f);
    P.grindEntry = comboAdd(GRINDS[kind].name, GRINDS[kind].base, GRINDS[kind].rate, true);
    P.flipKind = FK_NONE;
    P.pos = lerpv(r.a, r.b, t);
    P.squash = 0.5f;
    if (r.kind == RK_METAL) emitSparks(P.pos, dH * P.railDir, 10); else emitDust(P.pos, 5, 0.6f);
}
// Looks for a rail under the board while airborne; grinding wins over landing.
static bool trySnapRail(const V3& from, const V3& to) {
    if (P.anyRailCool > 0) return false;
    if (P.flipKind != FK_NONE && P.flipT < P.flipDur * 0.85f) return false;
    float hs = len2D(P.vel.x, P.vel.z);
    int best = -1;
    float bestT = 0, bestD = 1e9f;
    for (size_t i = 0; i < g_rails.size(); ++i) {
        if ((int)i == P.lastRail && P.railCool > 0) continue;
        const Rail& r = g_rails[i];
        float t = railParam(r, to.x, to.z);
        if (t <= 0.002f || t >= 0.998f) continue;
        V3 c = lerpv(r.a, r.b, t);
        float dxz = len2D(to.x - c.x, to.z - c.z), tol = r.kind == RK_CURB ? 0.3f : 0.42f;
        if (dxz > tol || to.y > c.y + 0.3f || to.y < c.y - 0.35f || from.y < c.y - 0.12f) continue;
        // Rolling off a ledge (no ollie) must not snap onto that ledge's own edge.
        if (!P.fromRail && P.airTime < 0.15f && len2D(P.takeoff.x - c.x, P.takeoff.z - c.z) < 0.7f && fabsf(P.takeoff.y - c.y) < 0.1f)
            continue;
        if (r.kind == RK_CURB) {
            V3 d = normalize(r.b - r.a);
            if (hs < 2.0f || fabsf(P.vel.x * d.x + P.vel.z * d.z) < 0.88f * hs) continue;  // curbs: only along them
        }
        if (dxz < bestD) { bestD = dxz; best = (int)i; bestT = t; }
    }
    if (best < 0) return false;
    startGrind(best, bestT);
    return true;
}
static void exitGrind(bool ollie, bool stall) {
    const Rail& r = g_rails[P.rail];
    V3 d = normalize(r.b - r.a) * P.railDir;
    comboFreeze(P.grindEntry);
    P.grindEntry = -1;
    if (strcmp(r.name, "Banks Handrail") == 0 && d.y < 0 && P.grindDist > 3.0f) awardGap(G_HANDRAIL);
    V3 v = d * P.railSpeed;
    if (ollie) v.y = std::max(v.y, 0.0f) + lerpf(POP_MIN, POP_MAX, P.charge) * 0.9f;
    else if (stall) { v = v + V3(-d.z, 0, d.x) * (P.bal >= 0 ? 1.2f : -1.2f); v.y = 1.5f; }
    else v.y += 1.2f;
    P.lastRail = P.rail;
    P.rail = -1;
    P.railCool = 0.4f; P.anyRailCool = 0.12f;
    beginAir(v, true);
    P.yaw = yawFromDir(d.x, d.z);
    if (ollie) {
        P.popT = 0.18f;
        P.spinVel = ((kLeft() ? 1.0f : 0.0f) - (kRight() ? 1.0f : 0.0f)) * 3.0f;
    }
}
static void switchGrind(int kind) {
    if (kind == P.grindKind) return;
    comboFreeze(P.grindEntry);
    P.grindKind = kind;
    P.grindEntry = comboAdd(GRINDS[kind].name, GRINDS[kind].base, GRINDS[kind].rate, true);
    P.bal *= 0.5f; P.balVel *= 0.5f;
    if (g_rails[P.rail].kind == RK_METAL) emitSparks(P.pos, fwdFromYaw(P.yaw), 6);
}
static void stepGrind(float dt) {
    const Rail& r = g_rails[P.rail];
    V3 d = normalize(r.b - r.a) * P.railDir;
    float L = std::max(0.1f, railLength(r));
    if (pressed('j')) switchGrind(GK_CROOK);
    if (pressed('k')) switchGrind(GK_SMITH);
    if (pressed('l')) switchGrind(GK_FEEBLE);
    if (pressed('u')) switchGrind(GK_NOSESLIDE);
    if (pressed('i')) switchGrind(GK_TAILSLIDE);
    if (pressed('w') || pressed(KEY_UP)) switchGrind(GRINDS[P.grindKind].slide ? GK_NOSESLIDE : GK_NOSE);
    if (pressed('s') || pressed(KEY_DOWN)) switchGrind(GRINDS[P.grindKind].slide ? GK_TAILSLIDE : GK_50);
    P.railSpeed += (-GRAVITY * 0.6f * d.y - 0.6f) * dt;
    float adv = P.railSpeed * dt;
    P.railT += P.railDir * adv / L;
    P.grindTime += dt;
    P.grindDist += adv;
    // Balance: the needle drifts away from centre faster the longer you grind; A/D push it back.
    float inst = 1.2f + 0.35f * P.grindTime, in = (kRight() ? 1.0f : 0.0f) - (kLeft() ? 1.0f : 0.0f);
    P.balVel += (P.bal * inst + frange(-0.7f, 0.7f) + in * 3.2f) * dt;
    P.balVel *= 1.0f / (1.0f + 1.5f * dt);
    P.bal += P.balVel * dt;
    if (fabsf(P.bal) > 1.0f) {
        V3 side = V3(-d.z, 0, d.x) * (P.bal > 0 ? 2.0f : -2.0f);
        bail("Lost your balance!", false, d * (P.railSpeed * 0.5f) + side + V3(0, 1.0f, 0));
        return;
    }
    if (P.railT >= 1.0f || P.railT <= 0.0f) {
        P.railT = clampf(P.railT, 0, 1);
        P.pos = lerpv(r.a, r.b, P.railT);
        exitGrind(false, false);
        return;
    }
    if (P.railSpeed < 1.0f) { exitGrind(false, true); return; }
    P.pos = lerpv(r.a, r.b, P.railT);
    P.vel = d * P.railSpeed;
    P.yaw = yawFromDir(d.x, d.z);
    if (strcmp(r.name, "Hudson Rail") == 0 && P.grindDist > 15.0f) awardGap(G_HUDSON);
    P.sparkAcc += dt * (r.kind == RK_METAL ? 45.0f : 12.0f);
    while (P.sparkAcc >= 1) {
        P.sparkAcc -= 1;
        if (r.kind == RK_METAL) emitSparks(P.pos, d, 1);
        else emitDust(P.pos, 1, 0.25f);
    }
}

// ---- Manuals ------------------------------------------------------------------------
static void startManual() {
    P.manual = true;
    P.noseManual = kFwd();
    P.bal = frange(-0.1f, 0.1f); P.balVel = 0;
    P.manualTime = 0;
    P.bankPending = false;
    P.manualEntry = comboAdd(P.noseManual ? "Nose Manual" : "Manual", P.noseManual ? 150.0f : 100.0f,
                             P.noseManual ? 180.0f : 150.0f, true);
}
static void endManual() {
    comboFreeze(P.manualEntry);
    P.manual = false;
    comboBank();
}
static void stepManualBalance(float dt) {
    P.manualTime += dt;
    // bal > 0: nose up (tipping back).  S lifts the nose, W pushes it down.
    float inst = 1.0f + 0.3f * P.manualTime, in = (kBack() ? 1.0f : 0.0f) - (kFwd() ? 1.0f : 0.0f);
    P.balVel += (P.bal * inst + frange(-0.6f, 0.6f) + in * 3.0f) * dt;
    P.balVel *= 1.0f / (1.0f + 1.5f * dt);
    P.bal += P.balVel * dt;
    if (fabsf(P.bal) > 1.0f) {
        V3 f = fwdFromYaw(P.yaw);
        bail(P.bal > 0 ? "Looped out of the manual!" : "Nose-dived!", false, f * (P.speed * 0.6f) + V3(0, 1.5f, 0));
    }
}

// ---- Bails ------------------------------------------------------------------------
static void bail(const char* why, bool toSafe, const V3& kick) {
    endGrab();
    comboLose(why);
    P.state = PS_BAIL;
    P.manual = false;
    P.bankPending = false;
    P.bailT = 0;
    P.bailSafe = toSafe;
    P.vel = kick;
    P.boardPos = P.pos + V3(0, 0.1f, 0);
    P.boardVel = kick * 1.2f + V3(frange(-1.0f, 1.0f), 2.5f, frange(-1.0f, 1.0f));
    P.boardSpin = frange(-12.0f, 12.0f);
    P.charge = 0; P.charging = false;
    P.flipKind = FK_NONE;
    P.rail = -1;
    P.grindEntry = P.manualEntry = -1;
}
static void respawn() {
    GroundHit g = queryGround(P.pos.x, P.pos.z, P.pos.y + 0.5f);
    int z = zoneAt(P.pos.x, P.pos.z);
    bool narrow = g.solid >= 0 && std::min(g_solids[g.solid].x1 - g_solids[g.solid].x0, g_solids[g.solid].z1 - g_solids[g.solid].z0) < 0.6f;
    bool hazard = P.bailSafe || narrow || z == ZN_STREET || z == ZN_RIVER || z == ZN_PIT || z == ZN_FOUNTAIN;
    V3 at = hazard ? P.safePos : V3(P.pos.x, g.h, P.pos.z);
    float yaw = hazard ? P.safeYaw : P.yaw;
    V3 sp = P.safePos;
    float sy = P.safeYaw;
    resetPlayer(false);
    float nx, nz;
    resolveWalls(at.x, at.z, at.y, at.y + STEP_UP, PLAYER_R, nx, nz);
    P.pos = at; P.yaw = yaw; P.safePos = sp; P.safeYaw = sy;
}
static void stepBail(float dt) {
    P.bailT += dt;
    P.vel.y -= GRAVITY * dt;
    V3 n = P.pos + P.vel * dt;
    float nx, nz;
    if (resolveWalls(n.x, n.z, n.y, n.y + STEP_UP, PLAYER_R, nx, nz)) {
        float vn = P.vel.x * nx + P.vel.z * nz;
        if (vn < 0) { P.vel.x -= nx * vn * 1.3f; P.vel.z -= nz * vn * 1.3f; }
    }
    int z = zoneAt(n.x, n.z);
    float wy = z == ZN_RIVER ? WATER_Y : (z == ZN_FOUNTAIN ? FOUNTAIN_WATER_Y : -100.0f);
    if (P.pos.y > wy && n.y <= wy) {
        emitSplash(V3(n.x, wy, n.z), 40, 1.6f, wy);
        P.vel *= 0.3f;
        P.bailSafe = true;
    }
    float floorY = z == ZN_RIVER ? WATER_Y - 0.6f : queryGround(n.x, n.z, n.y + 0.3f).h;
    if (n.y <= floorY) {
        n.y = floorY;
        P.vel.y = P.vel.y < -3.0f ? -P.vel.y * 0.25f : 0.0f;
        float k = 1.0f / (1.0f + (wy > -50 ? 8.0f : 5.0f) * dt);
        P.vel.x *= k; P.vel.z *= k;
    }
    P.pos = n;
    // The board tumbles away on its own.
    P.boardVel.y -= GRAVITY * dt;
    P.boardPos += P.boardVel * dt;
    resolveWalls(P.boardPos.x, P.boardPos.z, P.boardPos.y, P.boardPos.y + STEP_UP, 0.2f, nx, nz);
    int bz = zoneAt(P.boardPos.x, P.boardPos.z);
    float bf = bz == ZN_RIVER ? WATER_Y : (bz == ZN_FOUNTAIN ? FOUNTAIN_WATER_Y : queryGround(P.boardPos.x, P.boardPos.z, P.boardPos.y + 0.3f).h + 0.06f);
    if (P.boardPos.y < bf) {
        P.boardPos.y = bf;
        P.boardVel.y = P.boardVel.y < -2.0f ? -P.boardVel.y * 0.35f : 0.0f;
        P.boardVel.x *= 0.96f; P.boardVel.z *= 0.96f;
        P.boardSpin *= 0.9f;
    }
    if (P.bailT > 2.2f) respawn();
}

// ---- Gaps checked when an air ends (landing or snapping onto a rail) ----------------
static void airGaps(const V3& at, bool onRail) {
    const V3& t = P.takeoff;
    bool fromPlat = (!P.fromRail || t.y < PLATFORM_TOP + 0.6f) && t.y > PLATFORM_TOP - 0.05f &&
                    t.x > -46.5f && t.x < -29.5f && t.z > 17.5f && t.z < 34.5f;  // takeoff is sampled just past the edge
    if (fromPlat && !onRail && at.y < PLATFORM_TOP - 0.5f) {
        if (at.x > -26.4f && at.z > 21.0f && at.z < 31.0f) awardGap(G_STAIRS);
        else awardGap(G_DROP);
    }
    if (P.overFountain && zoneAt(t.x, t.z) != ZN_FOUNTAIN) {
        V3 a = t - FOUNTAIN_C, b = at - FOUNTAIN_C;
        if (a.x * b.x + a.z * b.z < 0 && len2D(at.x - t.x, at.z - t.z) > 6.0f) awardGap(G_FOUNTAIN);
    }
    if (P.overPit && !onRail && zoneAt(t.x, t.z) != ZN_PIT) awardGap(G_SUBWAY);
    if (P.overCar) awardGap(P.overTaxi ? G_TAXI : G_CAR);
}

static void land(const GroundHit& g) {
    float vy = P.vel.y, hs = len2D(P.vel.x, P.vel.z);
    float velYaw = hs > 2.0f ? yawFromDir(P.vel.x, P.vel.z) : P.yaw;
    V3 hv(P.vel.x, 0, P.vel.z);
    if (P.flipKind != FK_NONE && P.flipT < P.flipDur * 0.85f) { bail("Didn't finish the flip!", false, hv * 0.6f + V3(0, 1, 0)); return; }
    float rel = wrapPi(P.yaw + P.spin - velYaw);  // board heading vs. travel
    int halfTurns = (int)lroundf(rel / PI);
    float resid = fabsf(rel - halfTurns * PI);
    if (resid > radf(50)) { bail("Landed sideways!", false, hv * 0.6f + V3(0, 1, 0)); return; }
    applySpin((int)lroundf(fabsf(P.spin) / PI), P.spin);
    if (halfTurns != 0) P.fakie = !P.fakie;
    bool grabbing = P.grabKind >= 0;
    endGrab();
    P.state = PS_GROUND;
    P.yaw = velYaw;
    P.visYaw = wrapPi(rel - halfTurns * PI);
    P.spin = 0; P.spinVel = 0;
    P.flipKind = FK_NONE;
    P.speed = std::min(MAX_SPEED, hs * (1.0f - 0.35f * resid / radf(50)) * (grabbing ? 0.8f : 1.0f));
    if (resid > radf(28) || grabbing) toast("Sketchy!", Col(1.0f, 0.8f, 0.3f));
    P.squash = clampf(-vy / 12.0f, 0.2f, 1.0f);
    emitDust(P.pos, 4 + (int)std::min(10.0f, -vy), 0.8f);
    airGaps(P.pos, false);
    if (std::find(g_carRoofSolids.begin(), g_carRoofSolids.end(), g.solid) != g_carRoofSolids.end()) awardGap(G_SURF);
    if (g.solid == g_manualPad && g_manualPad >= 0 && P.takeoffSolid == g_kicker1 && g_keys['m']) awardGap(G_KICKMAN);
    P.bankPending = !g_combo.empty();
    P.bankTimer = 0.3f;  // window to catch a manual and keep the combo going
}

static void stepGround(float dt) {
    if (!P.manual && g_keys['m'] && P.speed > 1.5f) startManual();
    if (P.manual) {
        if (!g_keys['m'] || P.speed < 1.0f) endManual();
        else {
            stepManualBalance(dt);
            if (P.state != PS_GROUND) return;
        }
    }
    if (P.bankPending && (P.bankTimer -= dt) <= 0) { P.bankPending = false; comboBank(); }

    V3 f = fwdFromYaw(P.yaw);
    GroundHit g = queryGround(P.pos.x, P.pos.z, P.pos.y + 0.05f);
    P.groundSolid = g.solid;
    float slope = g.gx * f.x + g.gz * f.z;  // rise per metre along travel
    P.speed -= GRAVITY * 0.55f * slope / sqrtf(1 + slope * slope) * dt;
    if (P.speed < 0) { P.speed = -P.speed; P.yaw = wrapPi(P.yaw + PI); P.fakie = !P.fakie; slope = -slope; }
    bool canPush = !P.manual && !P.charging;
    if (canPush && kFwd() && P.speed < PUSH_MAX) P.speed = std::min(PUSH_MAX, P.speed + PUSH_ACC * dt);
    if (canPush && kBack()) P.speed = std::max(0.0f, P.speed - BRAKE * dt);
    P.speed -= ((P.manual ? 0.0f : ROLL_FRIC) + DRAG * P.speed * P.speed) * dt;  // manuals coast
    P.speed = clampf(P.speed, 0, MAX_SPEED);

    float turn = (kLeft() ? 1.0f : 0.0f) - (kRight() ? 1.0f : 0.0f);
    float rate = std::max(1.2f, 2.6f - 0.07f * P.speed) * (P.charging ? 1.3f : 1.0f) * (P.manual ? 0.6f : 1.0f);
    P.yaw = wrapPi(P.yaw + turn * rate * dt);
    f = fwdFromYaw(P.yaw);

    V3 n = P.pos + f * (P.speed * dt);
    float nx, nz;
    if (resolveWalls(n.x, n.z, P.pos.y, P.pos.y + STEP_UP, PLAYER_R, nx, nz)) {
        float vn = f.x * nx + f.z * nz;  // < 0: heading into the wall
        if (vn < -0.2f) {
            if (P.speed * -vn > 10.5f) { bail("Slammed into a wall!", false, V3(nx, 0, nz) * 2.5f + V3(0, 1.5f, 0)); return; }
            V3 t(f.x - nx * vn, 0, f.z - nz * vn);
            float tl = len2D(t.x, t.z);
            P.speed *= tl * (1.0f - 0.25f * -vn);
            if (tl > 0.05f) P.yaw = yawFromDir(t.x, t.z);
        }
    }
    GroundHit g2 = queryGround(n.x, n.z, P.pos.y + STEP_UP);
    float drop = P.pos.y - g2.h;
    if (drop > SNAP_DROP) {  // rolled off an edge or a kicker lip
        if (P.bankPending) { P.bankPending = false; comboBank(); }
        P.pos = n;
        beginAir(V3(f.x * P.speed, std::max(0.0f, slope) * P.speed, f.z * P.speed), false);
        return;
    }
    if (drop > 0.08f) { P.squash = std::max(P.squash, 0.3f); emitDust(V3(n.x, g2.h, n.z), 2, 0.4f); }  // stair clack
    P.pos = V3(n.x, g2.h, n.z);
    P.groundSolid = g2.solid;

    if (P.speed > 2.0f && inPuddle(P.pos.x, P.pos.z)) {  // splash through puddles
        for (P.splashAcc += dt * P.speed * 5.0f; P.splashAcc >= 1; P.splashAcc -= 1) {
            V3 side(f.z, 0, -f.x);
            Particle* q = emit(PK_WATER, P.pos - f * 0.3f + side * frange(-0.25f, 0.25f) + V3(0, 0.05f, 0),
                               side * frange(-1.5f, 1.5f) - f * (P.speed * 0.15f) + V3(0, frange(1.0f, 2.2f), 0), 1.0f,
                               frange(0.04f, 0.08f), Col(0.8f, 0.85f, 0.9f), 0.6f);
            q->floorY = P.pos.y;
        }
        if (frand() < dt * 8.0f) addRipple(P.pos.x, P.pos.y + 0.012f, P.pos.z, 1.0f, 1.0f);
    }
    int z = zoneAt(P.pos.x, P.pos.z);
    bool nearPit = P.pos.x > -27.5f && P.pos.x < -20.5f && P.pos.z > 8.0f && P.pos.z < 13.0f;
    if ((P.safeTimer -= dt) <= 0 && !P.manual && (z == ZN_SIDEWALK || z == ZN_PLAZA) && g2.solid < 0 && !nearPit &&
        fabsf(P.pos.x) < WORLD_X - 1.0f) {
        P.safePos = P.pos; P.safeYaw = P.yaw; P.safeTimer = 0.5f;
    }
}

static void stepAir(float dt) {
    P.airTime += dt;
    P.vel.y -= GRAVITY * dt;
    float in = (kLeft() ? 1.0f : 0.0f) - (kRight() ? 1.0f : 0.0f);
    if (in != 0) P.spinVel = clampf(P.spinVel + in * SPIN_ACC * dt, -SPIN_MAX, SPIN_MAX);
    else P.spinVel *= 1.0f / (1.0f + 6.0f * dt);
    P.spin += P.spinVel * dt;
    static const int flipKeys[5] = {'j', 'k', 'l', 'u', 'i'};
    for (int k = 0; k < 5; ++k) if (pressed(flipKeys[k])) pressFlip(k);
    if (P.flipKind != FK_NONE && (P.flipT += dt) >= P.flipDur) { P.flipKind = FK_NONE; P.flipT = 0; }
    if (g_keys['o'] && P.grabKind < 0 && P.flipKind == FK_NONE) startGrab();
    if (!g_keys['o'] && P.grabKind >= 0) endGrab();

    V3 n = P.pos + P.vel * dt;
    int zn = zoneAt(n.x, n.z);
    if (zn == ZN_FOUNTAIN) P.overFountain = true;
    if (zn == ZN_PIT) P.overPit = true;
    if (fabsf(n.x - HYDRANT_POS.x) < 1.2f && n.z > -8.4f && n.z < -4.5f && n.y < 2.2f) awardGap(G_HYDRANT);
    if (P.vel.y < 1.5f && trySnapRail(P.pos, n)) return;

    float nx, nz;
    if (resolveWalls(n.x, n.z, n.y, n.y + 0.1f, PLAYER_R, nx, nz)) {
        float vn = P.vel.x * nx + P.vel.z * nz;
        if (vn < 0) {
            if (-vn > 11.0f) { P.pos = n; bail("Slammed into a wall!", false, V3(nx, 0, nz) * 2.5f); return; }
            P.vel.x -= nx * vn; P.vel.z -= nz * vn;
        }
    }
    if (zn == ZN_FOUNTAIN && n.y < FOUNTAIN_WATER_Y + 0.05f) {
        P.pos = n;
        emitSplash(V3(n.x, FOUNTAIN_WATER_Y, n.z), 45, 1.6f, FOUNTAIN_WATER_Y);
        bail("Took a dip in the fountain!", true, P.vel * 0.15f);
        return;
    }
    if (zn == ZN_RIVER && n.y < 0.0f) { P.pos = n; bail("Went swimming in the Hudson!", true, P.vel * 0.5f); return; }
    if (zn == ZN_PIT && n.y < -0.6f) { P.pos = n; bail("Fell down the subway stairs!", true, P.vel * 0.3f); return; }
    GroundHit g = queryGround(n.x, n.z, std::max(P.pos.y, n.y) + 0.05f);
    if (n.y <= g.h) {
        P.pos = V3(n.x, g.h, n.z);
        P.groundSolid = g.solid;
        land(g);
        return;
    }
    P.pos = n;
}

// ---- Collisions with traffic and pedestrians -------------------------------------
static void interact() {
    if (P.state == PS_BAIL) return;
    float hs = len2D(P.vel.x, P.vel.z);
    for (Car& c : g_cars) {
        float dx = P.pos.x - c.x, dz = P.pos.z - c.z;
        if (fabsf(dx) > 2.25f + PLAYER_R || fabsf(dz) > 0.9f + PLAYER_R) continue;
        float roof = fabsf(dx - (c.dir > 0 ? -0.07f : 0.07f)) < 1.1f ? 1.45f : 0.95f;
        if (P.state == PS_AIR && P.pos.y > roof - 0.05f) {
            P.overCar = true;
            if (c.col == 0) P.overTaxi = true;
            continue;
        }
        if (c.speed > 2.0f) {
            V3 kick(c.dir * (c.speed * 0.8f + 2.0f), 3.5f, (dz >= 0 ? 1.0f : -1.0f) * 3.0f);
            bail(c.col == 0 ? "Hit by a cab!" : "Hit by a car!", true, kick);
            c.honkT = 2.5f; c.line = 4;
            return;
        }
        float px = 2.25f + PLAYER_R - fabsf(dx), pz = 0.9f + PLAYER_R - fabsf(dz);  // stopped car: solid
        if (px < pz) P.pos.x += dx >= 0 ? px : -px;
        else P.pos.z += dz >= 0 ? pz : -pz;
        if (P.state == PS_GROUND) P.speed *= 0.5f;
    }
    for (Npc& n : g_npcs) {
        float dx = P.pos.x - n.px, dz = P.pos.z - n.pz, d = len2D(dx, dz);
        if (d > 0.55f || P.pos.y > CURB_H + 1.4f) continue;
        float k = 1.0f / std::max(d, 0.01f);
        if (n.stun <= 0) { n.stun = 1.4f; npcSay(n, BUMP_LINES[irange(0, 9)], 2.4f); }
        if (hs > 6.5f || P.state == PS_GRIND) {
            bail("Plowed into a pedestrian!", false, V3(dx * k * 2.0f, 1.5f, dz * k * 2.0f) + P.vel * 0.3f);
            return;
        }
        P.pos.x = n.px + dx * k * 0.56f;
        P.pos.z = n.pz + dz * k * 0.56f;
        if (P.state == PS_GROUND) P.speed *= 0.4f;
        else { P.vel.x *= 0.4f; P.vel.z *= 0.4f; }
    }
}

static void updatePlayerVisuals(float dt) {
    float crouchT = 0.12f;
    if (P.charging) crouchT = 0.35f + 0.45f * P.charge;
    else if (P.state == PS_GRIND) crouchT = 0.45f;
    else if (P.state == PS_AIR) crouchT = P.grabKind >= 0 ? 0.9f : 0.55f;
    else if (P.manual) crouchT = 0.2f;
    P.crouch = lerpf(P.crouch, std::max(crouchT, P.squash), expK(14.0f, dt));
    P.squash = std::max(0.0f, P.squash - dt * 3.0f);
    bool pushing = P.state == PS_GROUND && kFwd() && !P.manual && !P.charging && P.speed < PUSH_MAX - 0.05f;
    if (P.pushT > 0 || pushing) {
        P.pushT += dt;
        if (P.pushT > 0.9f) P.pushT = pushing ? 0.0001f : 0.0f;
    }
    float pitchT = 0, rollT = 0, yawT = 0;
    if (P.manual) pitchT = (P.noseManual ? -0.28f : 0.28f) + 0.12f * P.bal;
    else if (P.state == PS_GRIND) {
        const GrindDef& gd = GRINDS[P.grindKind];
        pitchT = gd.pitch; rollT = gd.roll;
        yawT = gd.slide ? P.slideSign * PI * 0.5f : gd.yaw;
    }
    if (P.popT > 0) pitchT = 0.45f * (P.popT / 0.18f);
    P.popT = std::max(0.0f, P.popT - dt);
    P.boardPitch = lerpf(P.boardPitch, pitchT, expK(P.popT > 0 ? 30.0f : 12.0f, dt));
    P.boardRoll = lerpf(P.boardRoll, rollT, expK(12.0f, dt));
    P.visYaw = wrapPi(P.visYaw + wrapPi(yawT - P.visYaw) * expK(P.state == PS_GRIND ? 14.0f : 7.0f, dt));
    P.flipRoll = P.flipYaw = P.flipPitch = P.boardLift = 0;
    if (P.flipKind != FK_NONE && P.state == PS_AIR) {
        float u = clampf(P.flipT / P.flipDur, 0, 1), a = 2 * PI * u * P.flipCount;
        switch (P.flipKind) {
            case FK_KICK: P.flipRoll = a; break;
            case FK_HEEL: P.flipRoll = -a; break;
            case FK_SHOVE: P.flipYaw = PI * u * P.flipCount; break;
            case FK_TRE: P.flipRoll = a; P.flipYaw = a; break;
            case FK_IMPOSS: P.flipPitch = a; break;
            default: break;
        }
        P.boardLift = sinf(u * PI) * 0.22f;
    }
}

static void updateGame(float dt) {
    P.railCool = std::max(0.0f, P.railCool - dt);
    P.anyRailCool = std::max(0.0f, P.anyRailCool - dt);
    // Ollie: hold SPACE to crouch and charge, release to pop (also pops you off rails).
    bool canPop = P.state == PS_GROUND || P.state == PS_GRIND;
    if (canPop && !P.charging && (g_keys[' '] || pressed(' '))) { P.charging = true; P.charge = 0; }
    if (P.charging) {
        P.charge = std::min(1.0f, P.charge + dt / CHARGE_TIME);
        if (!g_keys[' ']) {
            P.charging = false;
            if (P.state == PS_GROUND) doOllie();
            else if (P.state == PS_GRIND) exitGrind(true, false);
            P.charge = 0;
        }
    }
    switch (P.state) {
        case PS_GROUND: stepGround(dt); break;
        case PS_AIR: stepAir(dt); break;
        case PS_GRIND: stepGrind(dt); break;
        default: stepBail(dt); break;
    }
    if (P.state == PS_GROUND) P.vel = fwdFromYaw(P.yaw) * P.speed;
    interact();
    comboTick(dt);
    updatePlayerVisuals(dt);
    for (Toast& t : g_toasts) t.t -= dt;
    g_toasts.erase(std::remove_if(g_toasts.begin(), g_toasts.end(), [](const Toast& t) { return t.t <= 0; }), g_toasts.end());
    g_popup.t = std::max(0.0f, g_popup.t - dt);
    bool inStreet = zoneAt(P.pos.x, P.pos.z) == ZN_STREET && P.pos.y < 1.0f;
    updateWorld(dt, P.pos, P.vel, inStreet);
}

// ----------------------------------------------------------------------------
//  Skater rendering
// ----------------------------------------------------------------------------
// Red hoodie, baggy jeans, chunky white shoes, black beanie: peak 2002.
static const Outfit SKATER_LOOK = {Col(0.78f, 0.12f, 0.1f), Col(0.22f, 0.32f, 0.52f), Col(0.86f, 0.66f, 0.5f), Col(0.12f, 0.08f, 0.05f),
                                   Col(0.95f, 0.95f, 0.95f), Col(0.1f, 0.1f, 0.12f), HAT_BEANIE, true, 1.35f};

// Rider/board frame: local +Z = the board end under the front foot, +X = toe side.
static float frameYaw() { return P.yaw + P.spin + P.visYaw + (P.fakie ? PI : 0.0f); }
static float riderYOffset() {
    if (P.state != PS_GRIND) return 0.0f;
    return GRINDS[P.grindKind].slide ? -0.075f : -0.035f;  // deck on the rail vs. trucks on the rail
}
static float boardPivotZ() { return (P.state == PS_AIR && P.popT <= 0) ? 0.0f : (P.boardPitch >= 0 ? -0.27f : 0.27f); }
static float boardRaise() {
    if (P.state != PS_AIR) return 0.0f;
    if (P.flipKind != FK_NONE) return P.boardLift;
    return P.grabKind >= 0 ? 0.3f : 0.0f;
}
static Frame playerFrame() { return makeFrame(P.pos + V3(0, riderYOffset(), 0), frameYaw(), 1.0f); }
// A point on the deck (local coords) after the deck pitches about the truck that stays down.
static V3 boardPoint(const Frame& F, float x, float y, float z) {
    float pz = boardPivotZ(), p = P.boardPitch, dz = z - pz;
    return F.at(x, y * cosf(p) + dz * sinf(p), pz + dz * cosf(p) - y * sinf(p));
}
static V3 fdir(const Frame& F, float phi) { return F.r * cosf(phi) + F.f * sinf(phi); }  // phi = 0: toe side
static V3 kneeIK(const V3& hip, const V3& foot, const V3& bend) {
    const float L = 0.45f;
    float h = len(foot - hip) * 0.5f;
    return (hip + foot) * 0.5f + bend * sqrtf(std::max(0.0f, L * L - h * h));
}

static Pose skaterPose(const Frame& F) {
    Pose S;
    const float deck = 0.09f, c = P.crouch;
    const V3 up(0, 1, 0);
    bool push = P.pushT > 0 && P.state == PS_GROUND;
    float u = clampf(P.pushT / 0.9f, 0, 1), pa = push ? sinf(u * PI) : 0.0f;
    float lift = boardRaise() + ((P.state == PS_AIR && P.flipKind != FK_NONE) ? 0.1f : 0.0f);
    S.footL = boardPoint(F, 0.02f, deck + lift, 0.24f);
    S.footR = boardPoint(F, 0.02f, deck + lift, -0.21f);
    if (push) {  // back foot kicks along the ground on the toe side
        static const float kt[6] = {0.0f, 0.18f, 0.3f, 0.72f, 0.88f, 1.0f};
        static const V3 kp[6] = {V3(0.02f, 0.09f, -0.21f), V3(0.16f, 0.22f, 0.05f), V3(0.22f, 0.0f, 0.12f),
                                 V3(0.22f, 0.0f, -0.5f), V3(0.12f, 0.2f, -0.38f), V3(0.02f, 0.09f, -0.21f)};
        int i = 0;
        while (i < 4 && u > kt[i + 1]) ++i;
        V3 l = lerpv(kp[i], kp[i + 1], smooth01((u - kt[i]) / (kt[i + 1] - kt[i])));
        S.footR = F.at(l.x, l.y, l.z);
    }
    float phi = 0.15f + 1.05f * pa;
    V3 d = fdir(F, phi), br = fdir(F, phi - PI * 0.5f);  // chest facing and the body's right
    float hipY = deck + 0.84f - 0.38f * c - 0.07f * pa;
    S.pelvis = F.at(-0.02f + 0.04f * c, hipY, 0.015f + 0.08f * pa);
    S.hipL = S.pelvis - br * 0.1f;
    S.hipR = S.pelvis + br * 0.1f;
    S.kneeL = kneeIK(S.hipL, S.footL, d);
    S.kneeR = kneeIK(S.hipR, S.footR, d);
    S.chest = S.pelvis + up * (0.46f - 0.06f * c) + d * (0.04f + 0.12f * c);
    V3 look = normalize(d * (push ? 0.2f : 0.55f) + F.f * (P.fakie ? -1.0f : 1.0f));
    S.head = S.chest + up * 0.26f + look * 0.03f;
    S.shL = S.chest - br * 0.19f;
    S.shR = S.chest + br * 0.19f;
    V3 tf = fwdFromYaw(P.yaw), tr(-tf.z, 0, tf.x), lean;
    if (P.state == PS_GRIND) lean = tr * (P.bal * 0.18f);
    else if (P.manual) lean = tf * (-P.bal * 0.12f);
    S.chest += lean; S.head += lean * 1.3f; S.shL += lean; S.shR += lean;
    if (P.state == PS_GRIND || P.manual) {  // arms out for balance
        S.elL = S.shL - br * 0.26f + up * (-0.04f + 0.1f * P.bal); S.hdL = S.shL - br * 0.5f + up * (0.2f * P.bal);
        S.elR = S.shR + br * 0.26f + up * (-0.04f - 0.1f * P.bal); S.hdR = S.shR + br * 0.5f + up * (-0.2f * P.bal);
    } else if (P.state == PS_AIR) {
        S.elL = S.shL - br * 0.2f - up * 0.1f + d * 0.05f; S.hdL = S.shL - br * 0.32f + up * 0.06f + d * 0.1f;
        S.elR = S.shR + br * 0.2f - up * 0.1f + d * 0.05f; S.hdR = S.shR + br * 0.32f + up * 0.06f + d * 0.1f;
    } else {
        float sw = push ? sinf(u * 2 * PI) * 0.12f : 0.0f;
        S.elL = S.shL - br * 0.06f - up * 0.26f + d * (0.05f + sw); S.hdL = S.elL - up * 0.24f + d * 0.08f - br * 0.02f;
        S.elR = S.shR + br * 0.06f - up * 0.26f + d * (0.05f - sw); S.hdR = S.elR - up * 0.24f + d * 0.08f + br * 0.02f;
    }
    if (P.grabKind >= 0 && P.state == PS_AIR) {
        if (P.grabKind == 0) { S.hdR = boardPoint(F, 0.1f, deck + lift, -0.02f); S.elR = (S.shR + S.hdR) * 0.5f + d * 0.12f; }
        else if (P.grabKind == 1) { S.hdL = boardPoint(F, 0.0f, deck + lift, 0.37f); S.elL = (S.shL + S.hdL) * 0.5f + d * 0.12f; }
        else { S.hdR = boardPoint(F, 0.0f, deck + lift, -0.37f); S.elR = (S.shR + S.hdR) * 0.5f + d * 0.12f; }
    }
    S.fwd = look;
    S.right = V3(-look.z, 0, look.x);
    S.footDirL = fdir(F, 0.55f + 0.9f * pa);
    S.footDirR = fdir(F, 0.12f);
    S.scale = 1.0f;
    return S;
}

// Standing figure that topples along the fall direction, arms flailing.
static Pose bailPose() {
    Frame F = makeFrame(P.pos, P.yaw, 1.0f);
    Pose S = walkPose(F, 0, 0);
    float t = P.bailT;
    S.elL = F.at(-0.35f, 1.35f + 0.1f * sinf(t * 14), 0.1f); S.hdL = F.at(-0.55f, 1.5f + 0.15f * sinf(t * 15), 0.15f);
    S.elR = F.at(0.35f, 1.35f + 0.1f * cosf(t * 13), 0.1f); S.hdR = F.at(0.55f, 1.5f + 0.15f * cosf(t * 12), 0.15f);
    V3 hv(P.vel.x, 0, P.vel.z);
    V3 dir = len(hv) > 0.5f ? normalize(hv) : F.f;
    V3 axis = normalize(cross(V3(0, 1, 0), dir));
    float ang = smooth01(t / 0.45f) * 1.45f, ca = cosf(ang), sa = sinf(ang);
    V3 c = P.pos + V3(0, 0.12f, 0);
    auto rotV = [&](const V3& v) { return v * ca + cross(axis, v) * sa + axis * (dot(axis, v) * (1 - ca)); };
    V3* pts[15] = {&S.hipL, &S.hipR, &S.kneeL, &S.kneeR, &S.footL, &S.footR, &S.pelvis, &S.chest, &S.head,
                   &S.shL, &S.shR, &S.elL, &S.elR, &S.hdL, &S.hdR};
    for (V3* p : pts) *p = c + rotV(*p - c);
    S.fwd = rotV(S.fwd); S.right = rotV(S.right); S.footDirL = rotV(S.footDirL); S.footDirR = rotV(S.footDirR);
    return S;
}

static void drawBoardModel() {
    texOff();
    glc(Col(0.95f, 0.92f, 0.82f));
    for (float zt : {-0.27f, 0.27f})
        for (float xs : {-1.0f, 1.0f}) drawCyl(V3(xs * 0.07f, 0.03f, zt), V3(xs * 0.105f, 0.03f, zt), 0.03f, 0.03f, 10);
    glc(Col(0.72f, 0.74f, 0.78f));
    for (float zt : {-0.27f, 0.27f}) {
        drawCyl(V3(-0.07f, 0.03f, zt), V3(0.07f, 0.03f, zt), 0.01f, 0.01f, 6, false);
        drawBox(-0.025f, 0.03f, zt - 0.035f, 0.025f, 0.075f, zt + 0.035f);
    }
    const Col grip(0.08f, 0.08f, 0.08f), graphic(0.95f, 0.75f, 0.1f);
    glc(grip); drawBox(-0.1f, 0.083f, -0.3f, 0.1f, 0.09f, 0.3f);
    glc(graphic); drawBox(-0.1f, 0.075f, -0.3f, 0.1f, 0.083f, 0.3f);
    glc(Col(0.85f, 0.15f, 0.2f)); drawBox(-0.03f, 0.0745f, -0.3f, 0.03f, 0.076f, 0.3f);
    for (int s = -1; s <= 1; s += 2) {  // kicked nose and tail
        glPushMatrix();
        glTranslatef(0, 0.0825f, s * 0.3f);
        glRotatef(s * -20.0f, 1, 0, 0);
        float z0 = s > 0 ? 0.0f : -0.1f, z1 = s > 0 ? 0.1f : 0.0f;
        glc(grip); drawBox(-0.1f, 0.0005f, z0, 0.1f, 0.0075f, z1);
        glc(graphic); drawBox(-0.1f, -0.0075f, z0, 0.1f, 0.0005f, z1);
        glPopMatrix();
    }
}
static void drawBoardAt(const V3& o, float yaw, float pitch, float pivotZ, float lift, float roll, float flipYaw, float flipPitch) {
    glPushMatrix();
    glTranslatef(o.x, o.y, o.z);
    glRotatef(degf(yaw), 0, 1, 0);
    glTranslatef(0, 0, pivotZ);
    glRotatef(-degf(pitch), 1, 0, 0);
    glTranslatef(0, 0, -pivotZ);
    glTranslatef(0, lift + 0.06f, 0);  // flips spin about the deck centre
    glRotatef(degf(flipYaw), 0, 1, 0);
    glRotatef(degf(roll), 0, 0, 1);
    glRotatef(-degf(flipPitch), 1, 0, 0);
    glTranslatef(0, -0.06f, 0);
    drawBoardModel();
    glPopMatrix();
}
static float bailBoardRoll() {
    float r0 = P.boardSpin * std::min(P.bailT, 0.8f);
    if (P.bailT <= 0.8f) return r0;
    return lerpf(r0, roundf(r0 / PI) * PI, smooth01((P.bailT - 0.8f) / 0.3f));
}
static void drawPlayer() {
    if (P.state == PS_BAIL) {
        drawFigure(bailPose(), SKATER_LOOK);
        drawBoardAt(P.boardPos - V3(0, 0.06f, 0), P.yaw + P.boardSpin * 0.5f * std::min(P.bailT, 0.8f), 0, 0, 0, bailBoardRoll(), 0, 0);
        return;
    }
    Frame F = playerFrame();
    drawFigure(skaterPose(F), SKATER_LOOK);
    drawBoardAt(F.o, frameYaw(), P.boardPitch, boardPivotZ(), boardRaise(), P.boardRoll + P.flipRoll, P.flipYaw, P.flipPitch);
}
static void drawPlayerShadow() {
    Pose S;
    V3 nose, tail;
    if (P.state == PS_BAIL) {
        S = bailPose();
        V3 f = fwdFromYaw(P.yaw + P.boardSpin * 0.5f * std::min(P.bailT, 0.8f));
        nose = P.boardPos + f * 0.4f;
        tail = P.boardPos - f * 0.4f;
    } else {
        Frame F = playerFrame();
        S = skaterPose(F);
        float r = boardRaise();
        nose = boardPoint(F, 0, 0.06f + r, 0.4f);
        tail = boardPoint(F, 0, 0.06f + r, -0.4f);
    }
    figureShadow(S, 0, true);
    glBegin(GL_QUADS);
    shadowSeg(nose, tail, 0.1f, 0, true);
    glEnd();
}

// ----------------------------------------------------------------------------
//  Camera
// ----------------------------------------------------------------------------
struct Camera { V3 pos, look; float yaw; bool init; };
static Camera g_cam = {V3(), V3(), 0.0f, false};
static V3 g_lastPlayerPos;

static void updateCamera(float dt) {
    bool snap = !g_cam.init || len(P.pos - g_lastPlayerPos) > 4.0f;  // first frame or respawn
    g_lastPlayerPos = P.pos;
    float hs = len2D(P.vel.x, P.vel.z), targetYaw = g_cam.yaw;
    if (P.state == PS_GROUND || P.state == PS_GRIND) targetYaw = P.yaw;
    else if (P.state == PS_AIR && hs > 1.0f) targetYaw = yawFromDir(P.vel.x, P.vel.z);
    if (snap) { targetYaw = P.yaw; g_cam.yaw = targetYaw; }
    float rate = P.state == PS_AIR ? 1.5f : (P.state == PS_GRIND ? 4.0f : 3.0f);
    g_cam.yaw = wrapPi(g_cam.yaw + wrapPi(targetYaw - g_cam.yaw) * expK(rate, dt));
    V3 f = fwdFromYaw(g_cam.yaw);
    float dist = 4.3f + std::min(P.speed, 14.0f) * 0.08f, height = 1.9f;
    V3 look = P.pos + V3(0, 1.1f, 0) + f * 1.2f;
    V3 pivot = P.pos + V3(0, 1.3f, 0), want = P.pos + V3(0, height, 0) - f * dist;
    for (int i = 1; i <= 24; ++i) {  // pull in when a building or the platform is behind us
        V3 p = lerpv(pivot, want, i / 24.0f);
        if (cameraBlocked(p)) { want = lerpv(pivot, want, (i - 1) / 24.0f); break; }
    }
    if (snap) { g_cam.pos = want; g_cam.look = look; g_cam.init = true; }
    float kh = expK(8.0f, dt), kv = expK(4.0f, dt);
    g_cam.pos.x = lerpf(g_cam.pos.x, want.x, kh);
    g_cam.pos.z = lerpf(g_cam.pos.z, want.z, kh);
    g_cam.pos.y = lerpf(g_cam.pos.y, want.y, kv);
    float floorY = std::max(queryGround(g_cam.pos.x, g_cam.pos.z, g_cam.pos.y + 0.5f).h, WATER_Y) + 0.4f;
    g_cam.pos.y = std::max(g_cam.pos.y, floorY);
    g_cam.look = lerpv(g_cam.look, look, expK(10.0f, dt));
}

// ----------------------------------------------------------------------------
//  Sky
// ----------------------------------------------------------------------------
static Col skyAt(float elev) {
    float t = powf(clampf(sinf(std::max(elev, 0.0f)), 0, 1), 0.55f);
    return Col(lerpf(SKY_HORIZON.r, SKY_ZENITH.r, t), lerpf(SKY_HORIZON.g, SKY_ZENITH.g, t), lerpf(SKY_HORIZON.b, SKY_ZENITH.b, t));
}
static void billboard(const V3& c, const V3& r, const V3& u, float s) {
    glTexCoord2f(0, 0); vtx(c - r * s - u * s);
    glTexCoord2f(1, 0); vtx(c + r * s - u * s);
    glTexCoord2f(1, 1); vtx(c + r * s + u * s);
    glTexCoord2f(0, 1); vtx(c - r * s + u * s);
}
static void drawSky(const V3& cam, const V3& camR, const V3& camU) {
    glDisable(GL_LIGHTING); glDisable(GL_FOG); glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE);
    texOff();
    const float R = 900.0f, el[9] = {-0.3f, 0.0f, 0.04f, 0.1f, 0.2f, 0.38f, 0.7f, 1.1f, 1.5707f};
    for (int i = 0; i < 8; ++i) {
        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= 32; ++j) {
            float a = 2 * PI * j / 32;
            for (int k = 1; k >= 0; --k) {
                float e = el[i + k];
                glc(skyAt(e));
                glVertex3f(cam.x + cosf(a) * cosf(e) * R, cam.y + sinf(e) * R, cam.z + sinf(a) * cosf(e) * R);
            }
        }
        glEnd();
    }
    glEnable(GL_BLEND);
    texOn(g_texParticle);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);  // drifting cumulus
    for (int i = 0; i < 14; ++i) {
        float az = hash01(i, 1, 91) * 2 * PI + g_time * 0.002f, e = 0.08f + 0.28f * hash01(i, 2, 91);
        V3 c = cam + V3(cosf(az) * cosf(e), sinf(e), sinf(az) * cosf(e)) * (R * 0.9f);
        for (int k = 0; k < 5; ++k) {
            V3 o = camR * ((k - 2) * 26.0f + hash01(i, k, 92) * 12.0f) + camU * (hash01(i, k, 93) * 14.0f);
            glColor4f(1.0f, 0.97f, 0.93f, 0.45f);
            billboard(c + o, camR, camU, 30.0f + 18.0f * hash01(i, k, 94));
        }
    }
    glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBegin(GL_QUADS);  // sun and its glow
    V3 sc = cam + SUN_DIR * (R * 0.95f);
    glColor4f(1.0f, 0.85f, 0.6f, 0.35f); billboard(sc, camR, camU, 260.0f);
    glColor4f(1.0f, 0.92f, 0.75f, 0.7f); billboard(sc, camR, camU, 70.0f);
    glColor4f(1.0f, 1.0f, 0.95f, 1.0f); billboard(sc, camR, camU, 26.0f);
    glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    texOff();
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE); glEnable(GL_DEPTH_TEST); glEnable(GL_FOG); glEnable(GL_LIGHTING);
}

// ----------------------------------------------------------------------------
//  Frame rendering
// ----------------------------------------------------------------------------
static Mat4 g_viewProj;
static int g_winW = 1280, g_winH = 720;

static void initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);
    const GLfloat amb[4] = {0.42f, 0.45f, 0.52f, 1}, dif[4] = {0.8f, 0.7f, 0.55f, 1}, zero[4] = {0, 0, 0, 1};
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, zero);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_EXP2);
    glFogf(GL_FOG_DENSITY, 0.0042f);
    const GLfloat fc[4] = {SKY_HORIZON.r, SKY_HORIZON.g, SKY_HORIZON.b, 1};
    glFogfv(GL_FOG_COLOR, fc);
    glHint(GL_FOG_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(0x809D);  // GL_MULTISAMPLE (no-op if the window has no multisample buffer)
}

static void renderScene() {
    glViewport(0, 0, g_winW, g_winH);
    glClearColor(SKY_HORIZON.r, SKY_HORIZON.g, SKY_HORIZON.b, 1);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    Mat4 proj = matPerspective(62.0f, (float)g_winW / std::max(1, g_winH), 0.2f, 1400.0f);
    Mat4 view = matLookAt(g_cam.pos, g_cam.look, V3(0, 1, 0));
    g_viewProj = matMul(proj, view);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj.m);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(view.m);
    V3 camR(view.m[0], view.m[4], view.m[8]), camU(view.m[1], view.m[5], view.m[9]);
    drawSky(g_cam.pos, camR, camU);
    const GLfloat lp[4] = {SUN_DIR.x, SUN_DIR.y, SUN_DIR.z, 0};
    glLightfv(GL_LIGHT0, GL_POSITION, lp);

    glCallList(g_worldList);
    drawRiver(g_cam.pos);
    drawFerry();
    drawCars();
    drawNpcs();
    drawPigeons();
    drawPlayer();

    // Shadows: one stencil-masked pass so overlaps never double-darken.
    glDisable(GL_LIGHTING);
    texOff();
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_EQUAL, 0, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2, -6);
    glColor4f(0.05f, 0.06f, 0.12f, 0.38f);
    glCallList(g_shadowList);
    drawCarShadows(g_cam.pos);
    drawNpcShadows();
    drawPlayerShadow();
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_STENCIL_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);

    drawWaterSurfaces(g_cam.pos);
    drawParticles(camR, camU);
}

// ----------------------------------------------------------------------------
//  HUD
// ----------------------------------------------------------------------------
static bool g_paused = false, g_helpPinned = false;
static float g_helpTimer = 14.0f;  // controls panel is shown for a while at start

static int textW(void* font, const std::string& s) {
    int w = 0;
    for (char c : s) w += glutBitmapWidth(font, c);
    return w;
}
static void textAt(void* font, float x, float y, const std::string& s, const Col& c, float a = 1.0f) {
    glColor4f(0, 0, 0, 0.75f * a);
    glRasterPos2f(x + 1, y - 1);
    for (char ch : s) glutBitmapCharacter(font, ch);
    glColor4f(c.r, c.g, c.b, a);
    glRasterPos2f(x, y);
    for (char ch : s) glutBitmapCharacter(font, ch);
}
static void rect(float x0, float y0, float x1, float y1, float r, float g, float b, float a) {
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0); glVertex2f(x1, y0); glVertex2f(x1, y1); glVertex2f(x0, y1);
    glEnd();
}
// Large outlined stroke text, centred on cx; shrinks to fit maxW.
static void strokeText(float cx, float y, const std::string& s, float h, const Col& c, float a, float maxW) {
    float w = 0;
    for (char ch : s) w += glutStrokeWidth(GLUT_STROKE_ROMAN, ch);
    float sc = h / 100.0f;
    if (w * sc > maxW) sc = maxW / w;
    w *= sc;
    glEnable(GL_LINE_SMOOTH);
    for (int pass = 0; pass < 2; ++pass) {
        float o = pass == 0 ? 2.5f : 1.0f;
        glLineWidth(pass == 0 ? 4.0f : 2.0f);
        if (pass == 0) glColor4f(0, 0, 0, 0.65f * a); else glColor4f(c.r, c.g, c.b, a);
        for (int ox = -1; ox <= 1; ++ox)
            for (int oy = -1; oy <= 1; ++oy) {
                glPushMatrix();
                glTranslatef(cx - w * 0.5f + ox * o, y + oy * o, 0);
                glScalef(sc, sc, 1);
                for (char ch : s) glutStrokeCharacter(GLUT_STROKE_ROMAN, ch);
                glPopMatrix();
            }
    }
    glLineWidth(1.0f);
    glDisable(GL_LINE_SMOOTH);
}
static std::string stateLabel() {
    switch (P.state) {
        case PS_GRIND: return std::string(GRINDS[P.grindKind].name) + " - " + g_rails[P.rail].name;
        case PS_AIR: return "Air   " + std::to_string((int)lroundf(fabsf(degf(P.spin)))) + " deg";
        case PS_BAIL: return "Bailed";
        default: return P.manual ? (P.noseManual ? "Nose Manual" : "Manual") : (P.speed < 0.3f ? "Standing" : "Rolling");
    }
}
static const char* HELP_LINES[] = {
    "W / Up  push          S / Down  brake",
    "A D / Left Right   turn on the ground, spin in the air",
    "SPACE   hold to crouch, release to ollie (hold longer = higher)",
    "J Kickflip   K Heelflip   L Pop Shove-It   U 360 Flip   I Impossible",
    "      tap the same flip again mid-air for Double / Triple",
    "O (hold)  grab: Indy, +W Nosegrab, +S Tailgrab  - let go before landing",
    "GRIND: land on any rail, ledge, bench or curb.  A/D keep balance.",
    "      hold W / S as you land: Nosegrind / 5-0.  Arrive sideways: Boardslide",
    "      J K L U I on a rail: Crooked, Smith, Feeble, Noseslide, Tailslide",
    "M (hold)  manual, W/S keep balance.  Manuals link tricks on the ground.",
    "COMBO = (sum of trick points) x (number of tricks).  Bail = lose it.",
    "Land spins within ~50 deg of a 180 and finish flips before touchdown.",
    "Find all the GAPS around the block for bonus points (list on the right).",
    "R reset    P pause    H show / hide this panel    ESC quit",
};

static void drawHud() {
    const float W = (float)g_winW, H = (float)g_winH;
    void* const F12 = GLUT_BITMAP_HELVETICA_12;
    void* const F18 = GLUT_BITMAP_HELVETICA_18;
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, W, 0, H, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG);
    texOff();
    glEnable(GL_BLEND);

    // Speech bubbles over pedestrians and honking cabs.
    auto bubble = [&](const V3& at, const char* text, float a) {
        float sx, sy;
        if (len(at - g_cam.pos) > 45.0f || !projectPoint(g_viewProj, at, g_winW, g_winH, sx, sy)) return;
        int w = textW(F12, text);
        float x0 = sx - w * 0.5f - 6, x1 = sx + w * 0.5f + 6, y0 = sy + 10, y1 = sy + 30;
        if (x0 < 2 || x1 > W - 2 || y0 < 2 || y1 > H - 2) return;
        rect(x0, y0, x1, y1, 1, 1, 1, 0.9f * a);
        glBegin(GL_TRIANGLES); glVertex2f(sx - 5, y0); glVertex2f(sx + 5, y0); glVertex2f(sx, sy + 2); glEnd();
        glColor4f(0.05f, 0.05f, 0.05f, a);
        glRasterPos2f(x0 + 6, y0 + 6);
        for (const char* p = text; *p; ++p) glutBitmapCharacter(F12, *p);
    };
    for (const Npc& n : g_npcs) if (n.bubbleT > 0) bubble(V3(n.px, CURB_H + 1.95f * n.scale, n.pz), n.bubble, std::min(1.0f, n.bubbleT * 3));
    for (const Car& c : g_cars) if (c.honkT > 0) bubble(V3(c.x, 1.9f, c.z), HONK_LINES[c.line], std::min(1.0f, c.honkT * 3));

    // Score (top left) and speed / state (top right).
    rect(12, H - 92, 300, H - 12, 0, 0, 0, 0.45f);
    textAt(F18, 22, H - 38, "SCORE  " + commas(g_score), Col(1.0f, 0.9f, 0.3f));
    textAt(F12, 22, H - 58, "Best combo  " + commas(g_bestCombo), Col(0.9f, 0.9f, 0.9f));
    textAt(F12, 22, H - 77, "Gaps found  " + std::to_string(gapsFound()) + " / " + std::to_string((int)G_COUNT), Col(0.5f, 1.0f, 0.6f));
    char buf[64];
    snprintf(buf, sizeof buf, "%d mph", (int)(len2D(P.vel.x, P.vel.z) * 2.237f + 0.5f));
    rect(W - 312, H - 74, W - 12, H - 12, 0, 0, 0, 0.45f);
    textAt(F18, W - 302, H - 38, buf, Col(1, 1, 1));
    textAt(F12, W - 200, H - 37, P.fakie ? "FAKIE" : "REGULAR", P.fakie ? Col(1.0f, 0.6f, 0.3f) : Col(0.7f, 0.85f, 1.0f));
    textAt(F12, W - 302, H - 60, stateLabel(), Col(0.85f, 0.85f, 0.85f));

    // Current combo (bottom centre).
    const float baseY = 66;
    if (!g_combo.empty()) {
        int n = comboMult();
        std::string names = comboText(90);
        void* nf = textW(F18, names) < W - 40 ? F18 : F12;
        textAt(nf, W * 0.5f - textW(nf, names) * 0.5f, baseY, names, Col(1, 1, 1));
        Col c = n >= 5 ? Col(1.0f, 0.5f, 0.2f) : (n >= 3 ? Col(1.0f, 0.85f, 0.2f) : Col(0.6f, 0.9f, 1.0f));
        float h = 28 + std::min(n, 10) * 2.0f;
        strokeText(W * 0.5f, baseY + 24, commas((long)(comboBase() + 0.5f)) + "  x  " + std::to_string(n), h, c, 1, W * 0.8f);
        if (n >= 3) strokeText(W * 0.5f, baseY + 30 + h * 1.15f, "COMBO!", 20, c, 0.55f + 0.45f * fabsf(sinf(g_time * 6)), W);
    }
    // Balance meters near the skater.
    if (P.state == PS_GRIND) {
        float cx = W * 0.5f, cy = H * 0.5f + 105;
        rect(cx - 110, cy - 6, cx + 110, cy + 6, 0, 0, 0, 0.55f);
        rect(cx - 32, cy - 6, cx + 32, cy + 6, 0.3f, 0.85f, 0.35f, 0.5f);
        float nx = cx + clampf(P.bal, -1, 1) * 108;
        Col nc = fabsf(P.bal) > 0.7f ? Col(1.0f, 0.25f, 0.2f) : Col(1, 1, 1);
        rect(nx - 3, cy - 13, nx + 3, cy + 13, nc.r, nc.g, nc.b, 1);
        textAt(F12, cx - 46, cy + 18, "A  balance  D", Col(0.9f, 0.9f, 0.9f), 0.8f);
    } else if (P.manual) {
        float cx = W * 0.5f + 95, cy = H * 0.5f - 40;
        rect(cx - 6, cy - 80, cx + 6, cy + 80, 0, 0, 0, 0.55f);
        rect(cx - 6, cy - 24, cx + 6, cy + 24, 0.3f, 0.85f, 0.35f, 0.5f);
        float ny = cy + clampf(P.bal, -1, 1) * 78;
        Col nc = fabsf(P.bal) > 0.7f ? Col(1.0f, 0.25f, 0.2f) : Col(1, 1, 1);
        rect(cx - 13, ny - 3, cx + 13, ny + 3, nc.r, nc.g, nc.b, 1);
        textAt(F12, cx + 12, cy + 70, "S", Col(0.9f, 0.9f, 0.9f), 0.8f);
        textAt(F12, cx + 12, cy - 78, "W", Col(0.9f, 0.9f, 0.9f), 0.8f);
    }
    if (P.charging) {  // ollie charge
        float cx = W * 0.5f - 95, cy = H * 0.5f - 40;
        rect(cx - 6, cy - 50, cx + 6, cy + 50, 0, 0, 0, 0.55f);
        rect(cx - 5, cy - 49, cx + 5, cy - 49 + 98 * P.charge, 1.0f, 0.8f - 0.5f * P.charge, 0.2f, 0.9f);
        textAt(F12, cx - 14, cy - 66, "POP", Col(1, 1, 1), 0.8f);
    }
    // Landing / bail feedback.
    if (g_popup.t > 0) {
        float age = g_popup.dur - g_popup.t, a = std::min(1.0f, g_popup.t / 0.5f);
        float pop = 1.0f + 0.35f * std::max(0.0f, 1.0f - age / 0.18f), y = H * 0.68f + age * 10.0f;
        strokeText(W * 0.5f, y, g_popup.big, 40 * pop, g_popup.col, a, W * 0.9f);
        if (!g_popup.small.empty()) {
            void* f = textW(F18, g_popup.small) < W - 40 ? F18 : F12;
            textAt(f, W * 0.5f - textW(f, g_popup.small) * 0.5f, y - 28, g_popup.small, Col(1, 1, 1), a);
        }
    }
    for (size_t i = 0; i < g_toasts.size(); ++i) {
        const Toast& t = g_toasts[g_toasts.size() - 1 - i];
        textAt(F18, W * 0.5f - textW(F18, t.text) * 0.5f, H - 40 - i * 26.0f, t.text, t.col, std::min(1.0f, t.t * 2));
    }
    // Help panel and gap checklist.
    if (g_helpPinned || g_helpTimer > 0) {
        const int n = sizeof(HELP_LINES) / sizeof(HELP_LINES[0]);
        float top = H - 104, a = g_helpPinned ? 1.0f : std::min(1.0f, g_helpTimer);
        rect(12, top - 34 - n * 18.0f, 492, top, 0, 0, 0, 0.55f * a);
        textAt(F18, 22, top - 24, "NYC SKATE '02  -  HOW TO PLAY", Col(1.0f, 0.85f, 0.25f), a);
        for (int i = 0; i < n; ++i) textAt(F12, 22, top - 46 - i * 18.0f, HELP_LINES[i], Col(0.95f, 0.95f, 0.95f), a);
        rect(W - 262, top - 34 - G_COUNT * 17.0f, W - 12, top + 18, 0, 0, 0, 0.55f * a);
        textAt(F18, W - 252, top - 6, "GAPS", Col(0.5f, 1.0f, 0.6f), a);
        for (int i = 0; i < G_COUNT; ++i) {
            std::string s = std::string(g_gapFound[i] ? "[x] " : "[  ] ") + GAPS[i].name + "  " + std::to_string(GAPS[i].pts);
            textAt(F12, W - 252, top - 28 - i * 17.0f, s, g_gapFound[i] ? Col(0.5f, 1.0f, 0.6f) : Col(0.8f, 0.8f, 0.8f), a);
        }
    }
    rect(0, 0, W, 30, 0, 0, 0, 0.55f);
    textAt(F12, 12, 10, "W/S push-brake   A/D turn-spin   SPACE hold+release: ollie   J K L U I flips   O grab   M manual   "
                        "land on rails to grind   R reset   P pause   H help", Col(0.92f, 0.92f, 0.92f));
    if (g_paused) {
        rect(0, 0, W, H, 0, 0, 0, 0.35f);
        strokeText(W * 0.5f, H * 0.5f, "PAUSED", 60, Col(1, 1, 1), 1, W);
    }
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING); glEnable(GL_FOG);
}

// ----------------------------------------------------------------------------
//  Main loop and input
// ----------------------------------------------------------------------------
static const double SIM_DT = 1.0 / 120.0;
static std::chrono::steady_clock::time_point g_last;
static double g_acc = 0;

static void display() {
    auto now = std::chrono::steady_clock::now();
    double frame = std::min(0.1, std::chrono::duration<double>(now - g_last).count());
    g_last = now;
    if (!g_paused) {
        g_acc += frame;
        int steps = 0;
        while (g_acc >= SIM_DT && steps < 24) {
            if (steps == 0) { g_stepPresses.swap(g_presses); g_presses.clear(); }
            else g_stepPresses.clear();
            updateGame((float)SIM_DT);
            g_acc -= SIM_DT;
            ++steps;
        }
        if (steps == 24) g_acc = 0;
        g_stepPresses.clear();
        updateCamera((float)frame);
        if (!g_helpPinned) g_helpTimer = std::max(0.0f, g_helpTimer - (float)frame);
    } else {
        g_presses.clear();
    }
    renderScene();
    drawHud();
    glutSwapBuffers();
}
static void toggleHelp() {
    bool visible = g_helpPinned || g_helpTimer > 0;
    g_helpPinned = !visible;
    g_helpTimer = 0;
}
static int normKey(unsigned char c) { return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c; }
static void onKeyDown(unsigned char c, int, int) {
    int k = normKey(c);
    if (k == 27) exit(0);
    if (k == 'p') g_paused = !g_paused;
    else if (k == 'h') toggleHelp();
    else if (k == 'r' && !g_paused) { resetPlayer(true); g_cam.init = false; }
    keyEvent(k, true);
}
static void onKeyUp(unsigned char c, int, int) { keyEvent(normKey(c), false); }
static int mapSpecial(int k) {
    switch (k) {
        case GLUT_KEY_LEFT: return KEY_LEFT;
        case GLUT_KEY_RIGHT: return KEY_RIGHT;
        case GLUT_KEY_UP: return KEY_UP;
        case GLUT_KEY_DOWN: return KEY_DOWN;
        case GLUT_KEY_F1: return KEY_F1;
        default: return -1;
    }
}
static void onSpecialDown(int k, int, int) {
    int m = mapSpecial(k);
    if (m == KEY_F1) toggleHelp();
    if (m >= 0) keyEvent(m, true);
}
static void onSpecialUp(int k, int, int) {
    int m = mapSpecial(k);
    if (m >= 0) keyEvent(m, false);
}
static void onReshape(int w, int h) { g_winW = std::max(1, w); g_winH = std::max(1, h); }
static void onIdle() { glutPostRedisplay(); }

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("NYC SKATE '02");
#ifdef __APPLE__
    GLint swapInterval = 1;  // vsync
    CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &swapInterval);
#endif
    glutIgnoreKeyRepeat(1);
    glutDisplayFunc(display);
    glutReshapeFunc(onReshape);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyDown);
    glutKeyboardUpFunc(onKeyUp);
    glutSpecialFunc(onSpecialDown);
    glutSpecialUpFunc(onSpecialUp);

    initGL();
    createTextures();
    buildWorld();
    initGameRefs();
    compileStatic();
    buildRiverGrid();
    g_parts.reserve(MAX_PARTS);
    initCars();
    initNpcs();
    initPigeons();
    resetPlayer(true);
    g_last = std::chrono::steady_clock::now();
    glutMainLoop();
    return 0;
}
