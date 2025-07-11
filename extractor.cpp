// extractor.cpp
// Annotate LaTeX regions in lecture slides and run a simulated OCR worker.
//
// CLI
// ----
// ./extractor <slides.pdf> [-o latex_regions]
//
// Key bindings inside the Slide Viewer window
// ------------------------------------------
// click-drag : draw a box
// u          : undo last box
// q          : save boxes & next slide
// b          : save boxes & back one slide
// c          : clear all boxes on current slide
// Esc        : quit program
//
// NOTE: This program depends on OpenCV (>= 4.0) and MuPDF.
//       Compile with something like:
//
//       g++ -std=c++23 extractor.cpp -o extractor \
//           $(pkg-config --cflags --libs opencv4) \
//           -lmupdf -lmupdf-third -pthread
//
//       The exact MuPDF linkage flags depend on your distribution.

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <mupdf/fitz.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// Viewer & key-binding configuration
struct ViewerConfig {
    static constexpr const char *WINDOW_NAME = "Slide Viewer";
    static constexpr int WINDOW_X = 100;
    static constexpr int WINDOW_Y = 100;
    static constexpr const char *TITLE_FMT = "%s - (%d / %d)"; // name, current, total
    static const inline cv::Scalar RECT_COLOR{0, 255, 0};      // BGR
    static constexpr int RECT_THICKNESS = 2;
};

struct Key {
    static constexpr int NEXT = 'q';  // save + next slide
    static constexpr int PREV = 'b';  // save + back one slide
    static constexpr int UNDO = 'u';  // undo last box
    static constexpr int CLEAR = 'c'; // clear all boxes
    static constexpr int ESC = 27;    // quit
};

// Cyclic list of dummy LaTeX snippets for the OCR worker
static const std::vector<std::string> LATEX_SNIPPETS = {
    R"(\hat{y}=\sigma(Wx+b))",
    R"(L=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2)",
    R"(p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)})",
    R"(\theta \leftarrow \theta-\eta\nabla_\theta L)",
    R"(q(z) \approx p(z \mid x))",
    R"(\mathrm{ELBO}=\mathbb{E}_{q}[\log p(x,z)]-\mathbb{E}_{q}[\log q(z)])",
    R"(K(x_i,x_j)=\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right))",
    R"(a^{(l)}=\mathrm{ReLU}(W^{(l)}a^{(l-1)}+b^{(l)}))",
    R"(\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}})",
    R"(f(x)=\mathrm{sign}(w^Tx+b))"};

static std::string next_latex_snippet() {
    static std::size_t index = 0;
    const std::string &s = LATEX_SNIPPETS[index % LATEX_SNIPPETS.size()];
    ++index;
    return s;
}

// Simulated OCR worker thread
void ocr_worker(const fs::path &folder, const std::atomic_bool &stop_flag) {
    std::cout << "[OCR] Worker started\n";
    while (!stop_flag.load()) {
        if (!fs::exists(folder)) {
            std::this_thread::sleep_for(2s);
            continue;
        }
        bool work_found = false;
        for (const auto &entry : fs::directory_iterator(folder)) {
            if (stop_flag.load()) break;
            if (entry.path().extension() != ".png") continue;
            fs::path tex_path = entry.path().replace_extension(".tex");
            if (fs::exists(tex_path)) continue; // already processed
            work_found = true;

            std::string latex = next_latex_snippet();
            std::cout << "[OCR] Processing " << entry.path().filename().string()
                      << " -> '" << latex << "'\n";

            // Simulate processing time (3 seconds in 30 × 100 ms steps)
            for (int i = 0; i < 30 && !stop_flag.load(); ++i) {
                std::this_thread::sleep_for(100ms);
            }
            if (stop_flag.load()) break;
            std::ofstream ofs(tex_path);
            ofs << latex << '\n';
            std::cout << "[OCR]   -> wrote " << tex_path.filename().string() << "\n";
        }
        if (!work_found) {
            std::this_thread::sleep_for(1s);
        }
    }
    std::cout << "[OCR] Worker shutting down\n";
}

// Bounding-box annotation helper class
class BoxDrawer {
public:
    BoxDrawer(const cv::Mat &img, int slide_num, int total_slides)
        : original_(img.clone()), slide_num_(slide_num), total_slides_(total_slides) {
        cv::namedWindow(ViewerConfig::WINDOW_NAME, cv::WINDOW_NORMAL);
        cv::moveWindow(ViewerConfig::WINDOW_NAME, ViewerConfig::WINDOW_X, ViewerConfig::WINDOW_Y);
        cv::setMouseCallback(ViewerConfig::WINDOW_NAME, &BoxDrawer::mouseCallback, this);
    }

    // Return value: "next", "back", or "quit"
    std::string run(std::vector<std::pair<cv::Point, cv::Point>> &out_boxes) {
        while (true) {
            cv::Mat frame = original_.clone();
            for (const auto &[p1, p2] : boxes_) {
                cv::rectangle(frame, p1, p2, ViewerConfig::RECT_COLOR, ViewerConfig::RECT_THICKNESS);
            }
            char title[128];
            std::snprintf(title, sizeof(title), ViewerConfig::TITLE_FMT,
                ViewerConfig::WINDOW_NAME, slide_num_, total_slides_);
            cv::setWindowTitle(ViewerConfig::WINDOW_NAME, title);
            cv::imshow(ViewerConfig::WINDOW_NAME, frame);
            int key = cv::waitKey(1);
            if (key == Key::NEXT) {
                out_boxes = boxes_;
                return "next";
            }
            if (key == Key::PREV) {
                out_boxes = boxes_;
                return "back";
            }
            if (key == Key::ESC) {
                out_boxes.clear();
                return "quit";
            }
            if (key == Key::UNDO && !boxes_.empty()) {
                boxes_.pop_back();
            }
            if (key == Key::CLEAR) {
                boxes_.clear();
            }
        }
    }

private:
    static void mouseCallback(int event, int x, int y, int /*flags*/, void *userdata) {
        auto *self = static_cast<BoxDrawer *>(userdata);
        if (event == cv::EVENT_LBUTTONDOWN) {
            self->start_ = {x, y};
        } else if (event == cv::EVENT_LBUTTONUP && self->start_.has_value()) {
            self->boxes_.emplace_back(*self->start_, cv::Point{x, y});
            self->start_.reset();
        }
    }

    cv::Mat original_;
    int slide_num_;
    int total_slides_;
    std::vector<std::pair<cv::Point, cv::Point>> boxes_;
    std::optional<cv::Point> start_;
};

// Save image crops for one slide
static void save_crops(const cv::Mat &img,
    const std::vector<std::pair<cv::Point, cv::Point>> &boxes,
    int slide_idx,
    const fs::path &out_dir) {
    int h = img.rows;
    int w = img.cols;

    int crop_idx = 1;
    for (const auto &[p1, p2] : boxes) {
        int x1 = std::clamp(std::min(p1.x, p2.x), 0, w);
        int x2 = std::clamp(std::max(p1.x, p2.x), 0, w);
        int y1 = std::clamp(std::min(p1.y, p2.y), 0, h);
        int y2 = std::clamp(std::max(p1.y, p2.y), 0, h);
        if (x2 - x1 == 0 || y2 - y1 == 0) continue; // empty crop

        cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        char fname[64];
        std::snprintf(fname, sizeof(fname), "slide_%03d_crop_%d.png", slide_idx + 1, crop_idx++);
        fs::path crop_path = out_dir / fname;
        cv::imwrite(crop_path.string(), crop);
        std::cout << "[GUI] Saved " << crop_path.filename().string() << "\n";
    }
}

// Annotate PDF deck & launch GUI
static void annotate_pdf(const fs::path &pdf_path, const fs::path &out_dir) {
    //  Open document
    fz_context *ctx = fz_new_context(nullptr, nullptr, FZ_STORE_DEFAULT);
    if (!ctx) throw std::runtime_error("Cannot create MuPDF context");
    fz_register_document_handlers(ctx);

    fz_document *doc = nullptr;
    try {
        doc = fz_open_document(ctx, pdf_path.string().c_str());
    } catch (...) {
        fz_drop_context(ctx);
        throw;
    }
    int page_count = fz_count_pages(ctx, doc);
    if (page_count <= 0) {
        fz_drop_document(ctx, doc);
        fz_drop_context(ctx);
        throw std::runtime_error("PDF contains no pages");
    }

    int slide_idx = 0;
    while (slide_idx >= 0 && slide_idx < page_count) {
        fz_page *page = fz_load_page(ctx, doc, slide_idx);
        fz_matrix mtx = fz_scale(200.0 / 72.0, 200.0 / 72.0); // 200 dpi render
        fz_rect bbox = fz_bound_page(ctx, page);
        fz_rect bounds;
        fz_transform_rect(&bounds, &bbox, &mtx);

        fz_pixmap *pix = fz_new_pixmap_with_bbox(ctx, fz_device_rgb(ctx), &bounds, nullptr, 0);
        fz_device *dev = fz_new_draw_device(ctx, pix);
        fz_run_page(ctx, page, dev, &mtx, nullptr);
        fz_close_device(ctx, dev);
        fz_drop_device(ctx, dev);

        int w = fz_pixmap_width(ctx, pix);
        int h = fz_pixmap_height(ctx, pix);
        const unsigned char *samples = fz_pixmap_samples(ctx, pix);

        // MuPDF gives BGRA; convert to BGR (drop alpha)
        cv::Mat img_rgba(h, w, CV_8UC4, const_cast<unsigned char *>(samples));
        cv::Mat img_bgr;
        cv::cvtColor(img_rgba, img_bgr, cv::COLOR_RGBA2BGR);

        BoxDrawer drawer(img_bgr, slide_idx + 1, page_count);
        std::vector<std::pair<cv::Point, cv::Point>> boxes;
        std::string action = drawer.run(boxes);

        if (action == "quit") {
            fz_drop_pixmap(ctx, pix);
            fz_drop_page(ctx, page);
            break;
        }
        if (!boxes.empty()) {
            save_crops(img_bgr, boxes, slide_idx, out_dir);
        }
        if (action == "back" && slide_idx > 0) {
            --slide_idx;
        } else if (action == "next") {
            ++slide_idx;
        }

        fz_drop_pixmap(ctx, pix);
        fz_drop_page(ctx, page);
    }

    cv::destroyAllWindows();
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
}

// Basic command-line parsing (one positional + optional -o/--out)
struct CmdLine {
    fs::path pdf = "slides.pdf";
    fs::path outdir = "latex_regions";
};

static CmdLine parse_arguments(int argc, char *argv[]) {
    CmdLine cl;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--out") {
            if (i + 1 < argc) {
                cl.outdir = argv[++i];
            } else {
                throw std::invalid_argument("Option '-o/--out' expects a directory");
            }
        } else {
            cl.pdf = arg;
        }
    }
    return cl;
}

// Program entry
int main(int argc, char *argv[]) {
    try {
        std::cout << "Annotate LaTeX regions in a PDF deck of slides.\n";
        CmdLine cmd = parse_arguments(argc, argv);
        cmd.pdf = fs::absolute(cmd.pdf);
        cmd.outdir = fs::absolute(cmd.outdir);

        if (!fs::exists(cmd.pdf)) {
            std::cerr << "PDF file not found: " << cmd.pdf << "\n";
            return 1;
        }
        fs::create_directories(cmd.outdir);

        std::atomic_bool stop_flag{false};
        std::thread worker(ocr_worker, cmd.outdir, std::ref(stop_flag));

        annotate_pdf(cmd.pdf, cmd.outdir);

        stop_flag.store(true);
        worker.join();
        std::cout << "All done. Bye!\n";
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
