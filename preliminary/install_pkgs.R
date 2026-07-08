options(repos = c(CRAN = "https://cloud.r-project.org"))
.libPaths("rlib")  # use the project-local library
dir.create("rlib", showWarnings = FALSE)
pkgs <- c("data.table","vegan","lingtypology")
need <- setdiff(pkgs, rownames(installed.packages()))
if (length(need)) install.packages(
  need, lib="rlib", dependencies=TRUE,
  Ncpus = max(1L, parallel::detectCores() - 1L)
)
