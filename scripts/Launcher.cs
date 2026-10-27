using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

static class Launcher {
    // Windows command-line quoting, including trailing backslashes.
    static string Quote(string arg) {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in arg) {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes);
            result.Append(c); slashes = 0;
        }
        result.Append('\\', slashes * 2); result.Append('"');
        return result.ToString();
    }
    [STAThread]
    static int Main(string[] args) {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, ".venv", "Scripts", "python.exe");
        try {
            if (!File.Exists(python))
                throw new Exception("Python runtime missing. See WINDOWS_ROCM.md to set up the .venv folder.");
            string arguments = Quote(Path.Combine(root, "scripts", "desktop_entry.py"));
            foreach (string arg in args) arguments += " " + Quote(arg);
            var info = new ProcessStartInfo(python, arguments) {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using (var process = Process.Start(info)) {
                process.WaitForExit();
                if (process.ExitCode != 0)
                    MessageBox.Show("FastMatch could not start or stopped unexpectedly. See logs\\fastmatch.log for details.", "FastMatch", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return process.ExitCode;
            }
        } catch (Exception exc) {
            MessageBox.Show(exc.Message, "FastMatch", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
