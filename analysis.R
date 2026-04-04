#!/usr/bin/env Rscript

# package setup

library(data.table)
library(ggplot2)
library(fixest)
library(broom)
library(glmnet)
library(Matrix)
library(patchwork)
library(scales)

set.seed(1234)

# threading and environment configuration

n_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1")))
if (is.na(n_cores) || n_cores < 1L) n_cores <- 1L

data.table::setDTthreads(1L)

# fixest is currently only permitting 1 thread in your environment, so set it explicitly
fixest_nthreads <- 1L
fixest::setFixest_nthreads(fixest_nthreads)

out_dir <- Sys.getenv("OUTPUT_DIR", unset = "artifacts/results_uriel")
data_dir <- Sys.getenv("DATA_DIR", unset = "data")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "plots"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tables"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "rdata"), showWarnings = FALSE, recursive = TRUE)

# input data

stex  <- fread(file.path(data_dir, "stex_with_uriel_distances.csv"))
toefl <- fread(file.path(data_dir, "toefl_with_uriel_distances.csv"))

# variable definitions

distance_vars <- c(
  "featural_dist",
  "script_dist",
  "geographic_dist",
  "genetic_dist",
  "syntactic_dist",
  "morphological_dist",
  "phonological_dist",
  "inventory_dist"
)

mechanistic_vars <- c(
  "syntactic_dist",
  "morphological_dist",
  "phonological_dist",
  "inventory_dist"
)

z_distance_vars <- paste0("z_", distance_vars)
z_mechanistic_vars <- paste0("z_", mechanistic_vars)

selected_plot_z_vars <- c(
  "z_syntactic_dist",
  "z_morphological_dist",
  "z_phonological_dist",
  "z_inventory_dist"
)

toefl_outcomes <- c("Reading", "Listening", "Speaking", "Writing", "Total")
stex_primary_outcomes <- c("Speaking")

glmnet_inner_folds_default <- suppressWarnings(as.integer(Sys.getenv("GLMNET_INNER_FOLDS", unset = "3")))
if (is.na(glmnet_inner_folds_default) || glmnet_inner_folds_default < 3L) {
  glmnet_inner_folds_default <- 3L
}

glmnet_nlambda_default <- suppressWarnings(as.integer(Sys.getenv("GLMNET_NLAMBDA", unset = "30")))
if (is.na(glmnet_nlambda_default) || glmnet_nlambda_default < 20L) {
  glmnet_nlambda_default <- 30L
}

# outlier and leverage filtering configuration

parse_bool_env <- function(x, default = FALSE) {
  val <- Sys.getenv(x, unset = if (default) "true" else "false")
  tolower(val) %in% c("1", "true", "t", "yes", "y")
}

outlier_filter_mode <- tolower(Sys.getenv("OUTLIER_FILTER_MODE", unset = "none"))
if (!outlier_filter_mode %in% c("none", "zscore", "leverage", "both")) {
  stop("OUTLIER_FILTER_MODE must be one of: none, zscore, leverage, both")
}

outlier_filter_scope <- tolower(Sys.getenv("OUTLIER_FILTER_SCOPE", unset = "none"))
if (!outlier_filter_scope %in% c("none", "toefl", "stex", "both")) {
  stop("OUTLIER_FILTER_SCOPE must be one of: none, toefl, stex, both")
}

outlier_z_threshold <- suppressWarnings(as.numeric(Sys.getenv("OUTLIER_Z_THRESHOLD", unset = "2.5")))
if (is.na(outlier_z_threshold) || outlier_z_threshold <= 0) {
  outlier_z_threshold <- 2.5
}

outlier_leverage_mult <- suppressWarnings(as.numeric(Sys.getenv("OUTLIER_LEVERAGE_MULT", unset = "2.5")))
if (is.na(outlier_leverage_mult) || outlier_leverage_mult <= 0) {
  outlier_leverage_mult <- 2.5
}

outlier_use_selected_only <- parse_bool_env("OUTLIER_USE_SELECTED_ONLY", default = TRUE)

# new toggles

stex_distance_standardization_level <- tolower(
  Sys.getenv("STEX_DISTANCE_STANDARDIZATION_LEVEL", unset = "pair")
)
if (!stex_distance_standardization_level %in% c("row", "pair")) {
  stop("STEX_DISTANCE_STANDARDIZATION_LEVEL must be one of: row, pair")
}

stex_scatter_weighted_smooth <- parse_bool_env(
  "STEX_SCATTER_WEIGHTED_SMOOTH",
  default = TRUE
)

stex_outlier_include_outcome <- parse_bool_env(
  "STEX_OUTLIER_INCLUDE_OUTCOME",
  default = FALSE
)

toefl_outlier_include_outcome <- parse_bool_env(
  "TOEFL_OUTLIER_INCLUDE_OUTCOME",
  default = TRUE
)

# helper functions

save_objects <- function(obj_names, filename) {
  save(
    list = obj_names,
    file = file.path(out_dir, "rdata", filename),
    envir = .GlobalEnv,
    compress = "xz"
  )
}

write_partial_dt <- function(dt, csv_name, rdata_name) {
  fwrite(dt, file.path(out_dir, "tables", csv_name))
  save(dt, file = file.path(out_dir, "rdata", rdata_name), compress = "xz")
}

plapply <- function(X, FUN, ..., mc.cores = n_cores) {
  if (.Platform$OS.type == "unix" && mc.cores > 1L) {
    parallel::mclapply(X, FUN, ..., mc.cores = mc.cores)
  } else {
    lapply(X, FUN, ...)
  }
}

standardize_cols <- function(dt, cols, prefix = "z_") {
  for (v in cols) {
    mu <- mean(dt[[v]], na.rm = TRUE)
    s  <- sd(dt[[v]], na.rm = TRUE)
    new_name <- paste0(prefix, v)
    if (is.na(s) || s == 0) {
      dt[, (new_name) := NA_real_]
    } else {
      dt[, (new_name) := (get(v) - mu) / s]
    }
  }
  invisible(dt)
}

rmse <- function(y, pred) sqrt(mean((y - pred)^2))
mae  <- function(y, pred) mean(abs(y - pred))

oos_r2 <- function(y, pred, train_mean) {
  denom <- sum((y - train_mean)^2)
  if (denom == 0) return(NA_real_)
  1 - sum((y - pred)^2) / denom
}

poisson_dev <- function(y, mu) {
  mu <- pmax(mu, 1e-8)
  term <- ifelse(y == 0, 0, y * log(y / mu))
  2 * mean(term - (y - mu))
}

mode_char <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

save_plot <- function(p, filename, width = 10, height = 7, dpi = 300) {
  ggsave(
    filename = file.path(out_dir, "plots", filename),
    plot = p,
    width = width,
    height = height,
    dpi = dpi
  )
}

make_row_vfolds <- function(n, v = 5, repeats = 1, seed = 1234) {
  folds <- list()
  idx <- 1L

  for (r in seq_len(repeats)) {
    set.seed(seed + r)
    ord <- sample.int(n)
    split_id <- rep(seq_len(v), length.out = n)

    for (k in seq_len(v)) {
      test_idx <- ord[split_id == k]
      train_idx <- setdiff(seq_len(n), test_idx)

      folds[[idx]] <- list(
        repeat_id = r,
        fold = k,
        train = train_idx,
        test = test_idx
      )
      idx <- idx + 1L
    }
  }

  folds
}

make_group_vfolds <- function(groups, v = 5, repeats = 1, seed = 1234) {
  groups <- as.character(groups)
  ug <- unique(groups)
  v_eff <- min(v, length(ug))
  folds <- list()
  idx <- 1L

  for (r in seq_len(repeats)) {
    set.seed(seed + r)
    ug_shuf <- sample(ug, length(ug), replace = FALSE)
    split_id <- rep(seq_len(v_eff), length.out = length(ug_shuf))

    for (k in seq_len(v_eff)) {
      test_groups <- ug_shuf[split_id == k]
      test_idx  <- which(groups %in% test_groups)
      train_idx <- setdiff(seq_along(groups), test_idx)

      if (length(test_idx) == 0L || length(train_idx) == 0L) next

      folds[[idx]] <- list(
        repeat_id = r,
        fold = k,
        train = train_idx,
        test = test_idx
      )
      idx <- idx + 1L
    }
  }

  folds
}

make_sparse_x <- function(data, rhs) {
  mm <- sparse.model.matrix(
    as.formula(paste0("~ ", rhs)),
    data = data
  )
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

tidy_lm <- function(fit, dataset, outcome, model_name, model_block) {
  td <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  gl <- as.data.table(broom::glance(fit))
  td[, `:=`(
    dataset = dataset,
    outcome = outcome,
    model_name = model_name,
    model_block = model_block,
    n = nobs(fit),
    r.squared = gl$r.squared[1],
    adj.r.squared = gl$adj.r.squared[1],
    AIC = AIC(fit),
    BIC = BIC(fit)
  )]
  td
}

tidy_fixest <- function(fit, dataset, outcome, model_name, model_block, family_name) {
  td <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  td[, `:=`(
    dataset = dataset,
    outcome = outcome,
    model_name = model_name,
    model_block = model_block,
    family_name = family_name,
    n = nobs(fit),
    AIC = suppressWarnings(tryCatch(AIC(fit), error = function(e) NA_real_)),
    BIC = suppressWarnings(tryCatch(BIC(fit), error = function(e) NA_real_))
  )]
  td
}

extract_fixest_fit_stats <- function(fit, dataset, outcome, model_name) {
  gl <- as.data.table(broom::glance(fit))

  data.table(
    dataset = dataset,
    outcome = outcome,
    model_name = model_name,
    n = gl$nobs[1],
    rmse = gl$sigma[1],
    within_r2 = gl$within.r.squared[1],
    within_adj_r2 = gl$adj.within.r.squared[1],
    AIC = gl$AIC[1],
    BIC = gl$BIC[1]
  )
}

evaluate_glmnet_cv <- function(data, outcome, specs, folds, family = c("gaussian", "poisson")) {
  family <- match.arg(family)
  y <- data[[outcome]]

  x_cache <- vector("list", length(specs))
  names(x_cache) <- vapply(specs, function(s) s$name, character(1))

  for (i in seq_along(specs)) {
    if (!identical(specs[[i]]$type, "baseline_mean")) {
      x_cache[[i]] <- make_sparse_x(data, specs[[i]]$rhs)
    }
  }

  eval_one_spec <- function(spec, x_mat = NULL) {
    fold_res <- plapply(folds, function(fd) {
      y_train <- y[fd$train]
      y_test  <- y[fd$test]

      fallback_pred <- rep(mean(y_train), length(y_test))

      if (identical(spec$type, "baseline_mean")) {
        pred <- fallback_pred
      } else {
        x_train <- x_mat[fd$train, , drop = FALSE]
        x_test  <- x_mat[fd$test,  , drop = FALSE]

        inner_folds <- min(glmnet_inner_folds_default, length(y_train))

        if (inner_folds < 3L || ncol(x_train) == 0L || length(unique(y_train)) <= 1L) {
          pred <- fallback_pred
        } else {
          cvfit <- tryCatch(
            cv.glmnet(
              x = x_train,
              y = y_train,
              family = family,
              alpha = 0,
              nfolds = inner_folds,
              nlambda = glmnet_nlambda_default,
              standardize = TRUE,
              type.measure = if (family == "gaussian") "mse" else "deviance"
            ),
            error = function(e) NULL
          )

          if (is.null(cvfit)) {
            pred <- fallback_pred
          } else {
            pred <- as.numeric(
              predict(cvfit, newx = x_test, s = "lambda.1se", type = "response")
            )
          }
        }
      }

      data.table(
        model_name = spec$name,
        repeat_id = fd$repeat_id,
        fold = fd$fold,
        n_test = length(y_test),
        rmse = rmse(y_test, pred),
        mae = mae(y_test, pred),
        oos_r2 = if (family == "gaussian") oos_r2(y_test, pred, mean(y_train)) else NA_real_,
        poisson_dev = if (family == "poisson") poisson_dev(y_test, pred) else NA_real_
      )
    })

    rbindlist(fold_res, use.names = TRUE, fill = TRUE)
  }

  all_res <- vector("list", length(specs))

  for (i in seq_along(specs)) {
    spec <- specs[[i]]
    if (identical(spec$type, "baseline_mean")) {
      all_res[[i]] <- eval_one_spec(spec, x_mat = NULL)
    } else {
      all_res[[i]] <- eval_one_spec(spec, x_mat = x_cache[[spec$name]])
    }
  }

  rbindlist(all_res, use.names = TRUE, fill = TRUE)
}

coef_forest_plot <- function(dt, title_text) {
  plot_dt <- copy(dt)
  plot_dt <- plot_dt[
    term != "(Intercept)" &
      !grepl("^Sex|^C::|^L1_code::|^L2_code::|^Family", term)
  ]

  ggplot(plot_dt, aes(x = estimate, y = reorder(term, estimate))) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_point(size = 1.8) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.15) +
    facet_wrap(~ outcome, scales = "free_y") +
    labs(
      title = title_text,
      x = "Estimate (95% CI)",
      y = NULL
    ) +
    theme_bw(base_size = 11)
}

perf_bar_plot <- function(dt, metric, title_text) {
  ggplot(dt, aes(x = reorder(model_name, get(metric)), y = get(metric))) +
    geom_col() +
    coord_flip() +
    facet_wrap(~ outcome, scales = "free_y") +
    labs(
      title = title_text,
      x = NULL,
      y = metric
    ) +
    theme_bw(base_size = 11)
}

flag_extreme_rows <- function(
  dt,
  id_col,
  x_cols,
  y_cols = NULL,
  mode = c("none", "zscore", "leverage", "both"),
  z_thresh = 2.5,
  leverage_mult = 2.5
) {
  mode <- match.arg(mode)

  if (!id_col %in% names(dt)) {
    stop("id_col not found in dt: ", id_col)
  }

  x_cols <- intersect(x_cols, names(dt))
  y_cols <- intersect(y_cols, names(dt))
  use_cols <- unique(c(x_cols, y_cols))

  out <- data.table(row_index = seq_len(nrow(dt)))
  out[, (id_col) := dt[[id_col]]]
  out[, `:=`(
    zscore_flag = FALSE,
    leverage_flag = FALSE,
    drop_flag = FALSE,
    max_abs_z = NA_real_,
    leverage = NA_real_,
    leverage_threshold = NA_real_,
    reason = "kept"
  )]

  if (mode == "none" || length(use_cols) == 0L || nrow(dt) == 0L) {
    out[, row_index := NULL]
    return(out)
  }

  cc <- complete.cases(dt[, ..use_cols])
  if (!any(cc)) {
    out[, row_index := NULL]
    return(out)
  }

  idx_cc <- which(cc)
  dcc <- copy(dt[idx_cc])

  if (mode %in% c("zscore", "both")) {
    z_list <- lapply(use_cols, function(v) {
      x <- dcc[[v]]
      s <- sd(x, na.rm = TRUE)
      if (is.na(s) || s == 0) {
        rep(0, length(x))
      } else {
        (x - mean(x, na.rm = TRUE)) / s
      }
    })

    z_mat <- do.call(cbind, z_list)
    if (is.null(dim(z_mat))) {
      z_mat <- matrix(z_mat, ncol = 1L)
    }

    max_abs_z <- apply(abs(z_mat), 1L, max, na.rm = TRUE)
    z_flag <- max_abs_z > z_thresh

    out[idx_cc, `:=`(
      zscore_flag = z_flag,
      max_abs_z = max_abs_z
    )]
  }

  if (mode %in% c("leverage", "both") && length(x_cols) > 0L) {
    mm <- model.matrix(
      as.formula(paste0("~ ", paste(x_cols, collapse = " + "))),
      data = dcc
    )

    n <- nrow(mm)
    p <- ncol(mm)

    if (n > 0L && p > 0L) {
      qx <- qr(mm)
      qmat <- qr.Q(qx)
      h <- rowSums(qmat^2)
      lev_thresh <- leverage_mult * p / n

      out[idx_cc, `:=`(
        leverage = h,
        leverage_threshold = lev_thresh,
        leverage_flag = h > lev_thresh
      )]
    }
  }

  out[, drop_flag := zscore_flag | leverage_flag]
  out[zscore_flag == TRUE  & leverage_flag == TRUE,  reason := "zscore+leverage"]
  out[zscore_flag == TRUE  & leverage_flag == FALSE, reason := "zscore"]
  out[zscore_flag == FALSE & leverage_flag == TRUE,  reason := "leverage"]
  out[drop_flag == FALSE, reason := "kept"]

  out[, row_index := NULL]
  out[]
}

# data preparation

stex[, `:=`(
  Sex = factor(Sex),
  C = factor(C),
  L1_code = factor(L1_code),
  L2_code = factor(L2_code),
  Family = factor(Family)
)]

toefl[, `:=`(
  L1_code = factor(L1_code),
  L2_code = factor(L2_code)
)]

# row-level standardization for source tables
# TOEFL always uses row-level data directly.
# STEX row-level z columns are kept for row-level predictive analyses.
standardize_cols(stex, distance_vars)
standardize_cols(toefl, distance_vars)

toefl[, toefl_row_id := .I]

stex_main <- copy(
  stex[
    !is.na(L2_code) &
      L2 != "Monolingual"
  ]
)

stex_main <- stex_main[
  complete.cases(stex_main[, ..distance_vars])
]

stex_main[, pair_id := factor(paste(L1_code, L2_code, sep = "__"))]

stex_pair <- stex_main[, .(
  n = .N,
  Speaking = mean(Speaking, na.rm = TRUE),
  new_feat = mean(new_feat, na.rm = TRUE),
  new_sounds = mean(new_sounds, na.rm = TRUE),
  AaA = mean(AaA, na.rm = TRUE),
  LoR = mean(LoR, na.rm = TRUE),
  Edu.day = mean(Edu.day, na.rm = TRUE),
  Enroll = mean(Enroll, na.rm = TRUE),
  prop_female = mean(Sex == "Female", na.rm = TRUE),
  C = factor(mode_char(as.character(C))),
  L1_code = factor(as.character(first(L1_code))),
  L2_code = factor(as.character(first(L2_code))),
  featural_dist = first(featural_dist),
  script_dist = first(script_dist),
  geographic_dist = first(geographic_dist),
  genetic_dist = first(genetic_dist),
  syntactic_dist = first(syntactic_dist),
  morphological_dist = first(morphological_dist),
  phonological_dist = first(phonological_dist),
  inventory_dist = first(inventory_dist),
  z_featural_dist = first(z_featural_dist),
  z_script_dist = first(z_script_dist),
  z_geographic_dist = first(z_geographic_dist),
  z_genetic_dist = first(z_genetic_dist),
  z_syntactic_dist = first(z_syntactic_dist),
  z_morphological_dist = first(z_morphological_dist),
  z_phonological_dist = first(z_phonological_dist),
  z_inventory_dist = first(z_inventory_dist)
), by = pair_id]

# if STEX_DISTANCE_STANDARDIZATION_LEVEL=pair, overwrite the z_* columns
# using pair-level distances instead of inherited row-level z-scores.
if (stex_distance_standardization_level == "pair") {
  standardize_cols(stex_pair, distance_vars)
}

# snapshots before optional filtering
# these let us write out the actual dropped records later

toefl_pre_filter <- copy(toefl)
stex_pair_pre_filter <- copy(stex_pair)
stex_main_pre_filter <- copy(stex_main)

# optional outlier and leverage filtering

filter_x_cols <- if (outlier_use_selected_only) selected_plot_z_vars else z_distance_vars

filter_summary <- list()

toefl_filter_log <- data.table(
  toefl_row_id = toefl$toefl_row_id,
  zscore_flag = FALSE,
  leverage_flag = FALSE,
  drop_flag = FALSE,
  max_abs_z = NA_real_,
  leverage = NA_real_,
  leverage_threshold = NA_real_,
  reason = "kept"
)

n_toefl_before <- nrow(toefl)

if (outlier_filter_mode != "none" && outlier_filter_scope %in% c("toefl", "both")) {
  toefl_filter_log <- flag_extreme_rows(
    dt = toefl,
    id_col = "toefl_row_id",
    x_cols = filter_x_cols,
    y_cols = if (toefl_outlier_include_outcome) "Total" else NULL,
    mode = outlier_filter_mode,
    z_thresh = outlier_z_threshold,
    leverage_mult = outlier_leverage_mult
  )

  flagged_toefl_ids <- toefl_filter_log[drop_flag == TRUE, toefl_row_id]
  toefl <- toefl[!toefl_row_id %in% flagged_toefl_ids]
}

filter_summary[[length(filter_summary) + 1L]] <- data.table(
  dataset = "toefl",
  filter_mode = outlier_filter_mode,
  filter_scope = outlier_filter_scope,
  z_threshold = outlier_z_threshold,
  leverage_mult = outlier_leverage_mult,
  selected_only = outlier_use_selected_only,
  include_outcome = toefl_outlier_include_outcome,
  n_before = n_toefl_before,
  n_after = nrow(toefl),
  n_removed = n_toefl_before - nrow(toefl)
)

stex_pair_filter_log <- data.table(
  pair_id = stex_pair$pair_id,
  zscore_flag = FALSE,
  leverage_flag = FALSE,
  drop_flag = FALSE,
  max_abs_z = NA_real_,
  leverage = NA_real_,
  leverage_threshold = NA_real_,
  reason = "kept"
)

n_stex_pair_before <- nrow(stex_pair)
n_stex_main_before <- nrow(stex_main)

if (outlier_filter_mode != "none" && outlier_filter_scope %in% c("stex", "both")) {
  stex_pair_filter_log <- flag_extreme_rows(
    dt = stex_pair,
    id_col = "pair_id",
    x_cols = filter_x_cols,
    y_cols = if (stex_outlier_include_outcome) "Speaking" else NULL,
    mode = outlier_filter_mode,
    z_thresh = outlier_z_threshold,
    leverage_mult = outlier_leverage_mult
  )

  flagged_pairs <- as.character(stex_pair_filter_log[drop_flag == TRUE, pair_id])

  stex_pair <- stex_pair[!as.character(pair_id) %in% flagged_pairs]
  stex_main <- stex_main[!as.character(pair_id) %in% flagged_pairs]
}

filter_summary[[length(filter_summary) + 1L]] <- data.table(
  dataset = "stex_pair",
  filter_mode = outlier_filter_mode,
  filter_scope = outlier_filter_scope,
  z_threshold = outlier_z_threshold,
  leverage_mult = outlier_leverage_mult,
  selected_only = outlier_use_selected_only,
  include_outcome = stex_outlier_include_outcome,
  n_before = n_stex_pair_before,
  n_after = nrow(stex_pair),
  n_removed = n_stex_pair_before - nrow(stex_pair)
)

filter_summary[[length(filter_summary) + 1L]] <- data.table(
  dataset = "stex_main",
  filter_mode = outlier_filter_mode,
  filter_scope = outlier_filter_scope,
  z_threshold = outlier_z_threshold,
  leverage_mult = outlier_leverage_mult,
  selected_only = outlier_use_selected_only,
  include_outcome = stex_outlier_include_outcome,
  n_before = n_stex_main_before,
  n_after = nrow(stex_main),
  n_removed = n_stex_main_before - nrow(stex_main)
)

filter_summary <- rbindlist(filter_summary, use.names = TRUE, fill = TRUE)

# full dropped-row reporting

toefl_dropped_ids <- toefl_filter_log[drop_flag == TRUE, toefl_row_id]
stex_dropped_pair_ids <- stex_pair_filter_log[drop_flag == TRUE, pair_id]

toefl_dropped_rows <- merge(
  toefl_pre_filter[toefl_row_id %in% toefl_dropped_ids],
  toefl_filter_log[drop_flag == TRUE],
  by = "toefl_row_id",
  all.x = TRUE,
  sort = FALSE
)

stex_pair_dropped_rows <- merge(
  stex_pair_pre_filter[pair_id %in% stex_dropped_pair_ids],
  stex_pair_filter_log[drop_flag == TRUE],
  by = "pair_id",
  all.x = TRUE,
  sort = FALSE
)

stex_main_dropped_rows <- merge(
  stex_main_pre_filter[pair_id %in% stex_dropped_pair_ids],
  stex_pair_filter_log[drop_flag == TRUE],
  by = "pair_id",
  all.x = TRUE,
  sort = FALSE
)

toefl_drop_reason_summary <- toefl_filter_log[
  drop_flag == TRUE, .N, by = .(reason)
][order(-N)]

stex_pair_drop_reason_summary <- stex_pair_filter_log[
  drop_flag == TRUE, .N, by = .(reason)
][order(-N)]

fwrite(filter_summary, file.path(out_dir, "tables", "00_filter_summary.csv"))
fwrite(toefl_filter_log, file.path(out_dir, "tables", "00_toefl_filter_log.csv"))
fwrite(stex_pair_filter_log, file.path(out_dir, "tables", "00_stex_pair_filter_log.csv"))

fwrite(toefl_dropped_rows, file.path(out_dir, "tables", "00a_toefl_dropped_rows.csv"))
fwrite(stex_pair_dropped_rows, file.path(out_dir, "tables", "00b_stex_pair_dropped_rows.csv"))
fwrite(stex_main_dropped_rows, file.path(out_dir, "tables", "00c_stex_main_rows_from_dropped_pairs.csv"))
fwrite(toefl_drop_reason_summary, file.path(out_dir, "tables", "00d_toefl_drop_reason_summary.csv"))
fwrite(stex_pair_drop_reason_summary, file.path(out_dir, "tables", "00e_stex_pair_drop_reason_summary.csv"))

save(
  filter_summary,
  toefl_filter_log,
  stex_pair_filter_log,
  file = file.path(out_dir, "rdata", "00_filter_artifacts.RData"),
  compress = "xz"
)

save(
  toefl_dropped_rows,
  stex_pair_dropped_rows,
  stex_main_dropped_rows,
  toefl_drop_reason_summary,
  stex_pair_drop_reason_summary,
  file = file.path(out_dir, "rdata", "00b_dropped_row_artifacts.RData"),
  compress = "xz"
)

# descriptive analyses

toefl_cor <- rbindlist(lapply(toefl_outcomes, function(y) {
  rbindlist(lapply(distance_vars, function(d) {
    data.table(
      dataset = "toefl",
      outcome = y,
      distance = d,
      cor = cor(toefl[[y]], toefl[[d]], use = "pairwise.complete.obs")
    )
  }))
}))

fwrite(toefl_cor, file.path(out_dir, "tables", "01_toefl_correlations.csv"))

stex_pair_cor <- rbindlist(lapply(stex_primary_outcomes, function(y) {
  rbindlist(lapply(distance_vars, function(d) {
    data.table(
      dataset = "stex",
      outcome = y,
      distance = d,
      cor = cor(stex_pair[[y]], stex_pair[[d]], use = "pairwise.complete.obs")
    )
  }))
}))

fwrite(stex_pair_cor, file.path(out_dir, "tables", "02_stex_pair_correlations.csv"))
fwrite(stex_pair, file.path(out_dir, "tables", "03_stex_pair_summary.csv"))

p_toefl_scatter <- ggplot(
  melt(
    copy(toefl),
    measure.vars = selected_plot_z_vars,
    variable.name = "distance",
    value.name = "z_distance"
  ),
  aes(x = z_distance, y = Total)
) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~ distance, scales = "free_x") +
  theme_bw(base_size = 11) +
  labs(
    title = "TOEFL Total vs selected URIEL distances",
    x = "Standardised URIEL distance",
    y = "TOEFL Total"
  )

save_plot(p_toefl_scatter, "01_toefl_total_scatter.png", width = 12, height = 7)

stex_pair_scatter_dt <- melt(
  copy(stex_pair),
  measure.vars = selected_plot_z_vars,
  variable.name = "distance",
  value.name = "z_distance"
)

p_stex_pair_scatter <- ggplot(
  stex_pair_scatter_dt,
  aes(x = z_distance, y = Speaking, size = n)
) +
  geom_point(alpha = 0.65)

if (stex_scatter_weighted_smooth) {
  p_stex_pair_scatter <- p_stex_pair_scatter +
    geom_smooth(aes(weight = n), method = "lm", se = TRUE)
} else {
  p_stex_pair_scatter <- p_stex_pair_scatter +
    geom_smooth(method = "lm", se = TRUE)
}

p_stex_pair_scatter <- p_stex_pair_scatter +
  facet_wrap(~ distance, scales = "free_x") +
  theme_bw(base_size = 11) +
  labs(
    title = "STEX pair-level mean Speaking vs selected URIEL distances",
    x = "Standardised URIEL distance",
    y = "Mean Speaking",
    size = "Pair size"
  )

save_plot(p_stex_pair_scatter, "02_stex_pair_speaking_scatter.png", width = 12, height = 7)

save_objects(
  c(
    "filter_summary",
    "toefl_filter_log",
    "stex_pair_filter_log",
    "toefl_dropped_rows",
    "stex_pair_dropped_rows",
    "stex_main_dropped_rows",
    "toefl_drop_reason_summary",
    "stex_pair_drop_reason_summary",
    "toefl_cor",
    "stex_pair_cor",
    "stex_pair",
    "p_toefl_scatter",
    "p_stex_pair_scatter"
  ),
  "01_descriptive_artifacts.RData"
)

# inferential analyses for TOEFL

run_toefl_models <- function(dat, outcome) {
  uni <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- lm(as.formula(paste(outcome, "~", zv)), data = dat)
    tidy_lm(fit, dataset = "toefl", outcome = outcome, model_name = zv, model_block = "single_distance")
  }), use.names = TRUE, fill = TRUE)

  mech_rhs <- paste(z_mechanistic_vars, collapse = " + ")
  fit_mech <- lm(as.formula(paste(outcome, "~", mech_rhs)), data = dat)
  mech <- tidy_lm(fit_mech, dataset = "toefl", outcome = outcome, model_name = "mechanistic_block", model_block = "mechanistic_block")

  all_rhs <- paste(z_distance_vars, collapse = " + ")
  fit_all <- lm(as.formula(paste(outcome, "~", all_rhs)), data = dat)
  all8 <- tidy_lm(fit_all, dataset = "toefl", outcome = outcome, model_name = "all8_block", model_block = "all8_block")

  rbindlist(list(uni, mech, all8), use.names = TRUE, fill = TRUE)
}

toefl_inf <- rbindlist(lapply(toefl_outcomes, function(y) run_toefl_models(toefl, y)), use.names = TRUE, fill = TRUE)
toefl_inf[term != "(Intercept)", p_adj_fdr := p.adjust(p.value, method = "fdr"), by = .(outcome)]

fwrite(toefl_inf, file.path(out_dir, "tables", "04_toefl_inferential_results.csv"))

p_toefl_coef <- coef_forest_plot(
  toefl_inf[model_block == "single_distance" & grepl("^z_", term)],
  "TOEFL: one-distance-at-a-time inferential models"
)
save_plot(p_toefl_coef, "03_toefl_inferential_coefficients.png", width = 14, height = 9)

# inferential analyses for STEX

run_stex_speaking_models <- function(dat) {
  outcome <- "Speaking"
  base_rhs <- "AaA + LoR + Edu.day + prop_female + Enroll"

  res <- list()

  uni <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- feols(
      as.formula(
        paste0(outcome, " ~ ", zv, " + ", base_rhs, " | C + L1_code + L2_code")
      ),
      data = dat,
      weights = ~n,
      vcov = "hetero",
      nthreads = fixest_nthreads
    )
    tidy_fixest(
      fit,
      dataset = "stex",
      outcome = outcome,
      model_name = zv,
      model_block = "single_distance",
      family_name = "gaussian_pair_weighted"
    )
  }), use.names = TRUE, fill = TRUE)
  res[[length(res) + 1L]] <- uni

  mech_rhs <- paste(c(z_mechanistic_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + ")
  fit_mech <- feols(
    as.formula(
      paste0(outcome, " ~ ", mech_rhs, " | C + L1_code + L2_code")
    ),
    data = dat,
    weights = ~n,
    vcov = "hetero",
    nthreads = fixest_nthreads
  )
  mech <- tidy_fixest(
    fit_mech,
    dataset = "stex",
    outcome = outcome,
    model_name = "mechanistic_block",
    model_block = "mechanistic_block",
    family_name = "gaussian_pair_weighted"
  )
  res[[length(res) + 1L]] <- mech

  all_rhs <- paste(c(z_distance_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + ")
  fit_all <- feols(
    as.formula(
      paste0(outcome, " ~ ", all_rhs, " | C + L1_code + L2_code")
    ),
    data = dat,
    weights = ~n,
    vcov = "hetero",
    nthreads = fixest_nthreads
  )
  all8 <- tidy_fixest(
    fit_all,
    dataset = "stex",
    outcome = outcome,
    model_name = "all8_block",
    model_block = "all8_block",
    family_name = "gaussian_pair_weighted"
  )
  res[[length(res) + 1L]] <- all8

  inter_aaa <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- feols(
      as.formula(
        paste0(outcome, " ~ ", zv, " * AaA + LoR + Edu.day + prop_female + Enroll | C + L1_code + L2_code")
      ),
      data = dat,
      weights = ~n,
      vcov = "hetero",
      nthreads = fixest_nthreads
    )
    tidy_fixest(
      fit,
      dataset = "stex",
      outcome = outcome,
      model_name = paste0(zv, "_x_AaA"),
      model_block = "interaction_AaA",
      family_name = "gaussian_pair_weighted"
    )
  }), use.names = TRUE, fill = TRUE)
  res[[length(res) + 1L]] <- inter_aaa

  inter_lor <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- feols(
      as.formula(
        paste0(outcome, " ~ ", zv, " * LoR + AaA + Edu.day + prop_female + Enroll | C + L1_code + L2_code")
      ),
      data = dat,
      weights = ~n,
      vcov = "hetero",
      nthreads = fixest_nthreads
    )
    tidy_fixest(
      fit,
      dataset = "stex",
      outcome = outcome,
      model_name = paste0(zv, "_x_LoR"),
      model_block = "interaction_LoR",
      family_name = "gaussian_pair_weighted"
    )
  }), use.names = TRUE, fill = TRUE)
  res[[length(res) + 1L]] <- inter_lor

  rbindlist(res, use.names = TRUE, fill = TRUE)
}

stex_inf <- run_stex_speaking_models(stex_pair)

stex_inf[term != "(Intercept)", p_adj_fdr := p.adjust(p.value, method = "fdr"), by = .(outcome, model_block)]

write_partial_dt(
  stex_inf,
  "05_stex_inferential_results.csv",
  "05_stex_inferential_results.RData"
)

p_stex_coef <- coef_forest_plot(
  stex_inf[
    outcome == "Speaking" &
      model_block == "single_distance" &
      grepl("^z_", term)
  ],
  "STEX speaking inferential models"
)
save_plot(p_stex_coef, "04_stex_inferential_coefficients.png", width = 14, height = 8)

fit_all8_clustered <- feols(
  Speaking ~ z_featural_dist + z_script_dist + z_geographic_dist + z_genetic_dist +
    z_syntactic_dist + z_morphological_dist + z_phonological_dist + z_inventory_dist +
    AaA + LoR + Edu.day + prop_female + Enroll | C + L1_code + L2_code,
  data = stex_pair,
  weights = ~n,
  cluster = ~pair_id,
  nthreads = fixest_nthreads
)

all8_clustered_tidy <- as.data.table(broom::tidy(fit_all8_clustered, conf.int = TRUE))
fwrite(all8_clustered_tidy, file.path(out_dir, "tables", "05b_stex_speaking_all8_clustered_check.csv"))
save(all8_clustered_tidy, file = file.path(out_dir, "rdata", "05b_stex_speaking_all8_clustered_check.RData"), compress = "xz")

stex_base_rhs <- "AaA + LoR + Edu.day + prop_female + Enroll"

fit_stex_base <- feols(
  as.formula(
    paste0("Speaking ~ ", stex_base_rhs, " | C + L1_code + L2_code")
  ),
  data = stex_pair,
  weights = ~n,
  vcov = "hetero",
  nthreads = fixest_nthreads
)

fit_stex_mech <- feols(
  as.formula(
    paste0(
      "Speaking ~ ",
      paste(c(z_mechanistic_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + "),
      " | C + L1_code + L2_code"
    )
  ),
  data = stex_pair,
  weights = ~n,
  vcov = "hetero",
  nthreads = fixest_nthreads
)

fit_stex_all8 <- feols(
  as.formula(
    paste0(
      "Speaking ~ ",
      paste(c(z_distance_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + "),
      " | C + L1_code + L2_code"
    )
  ),
  data = stex_pair,
  weights = ~n,
  vcov = "hetero",
  nthreads = fixest_nthreads
)

stex_model_comparison <- rbindlist(list(
  extract_fixest_fit_stats(fit_stex_base, "stex", "Speaking", "base"),
  extract_fixest_fit_stats(fit_stex_mech, "stex", "Speaking", "base_plus_mechanistic"),
  extract_fixest_fit_stats(fit_stex_all8, "stex", "Speaking", "base_plus_all8")
), use.names = TRUE, fill = TRUE)

fwrite(
  stex_model_comparison,
  file.path(out_dir, "tables", "05c_stex_speaking_model_comparison.csv")
)
save(
  stex_model_comparison,
  file = file.path(out_dir, "rdata", "05c_stex_speaking_model_comparison.RData"),
  compress = "xz"
)

stex_wald_mech <- data.table(
  model = "base_plus_mechanistic",
  test = "mechanistic_block",
  stat = unname(wald(fit_stex_mech, keep = "z_")$stat),
  p = unname(wald(fit_stex_mech, keep = "z_")$p)
)

stex_wald_all8 <- data.table(
  model = "base_plus_all8",
  test = "all8_block",
  stat = unname(wald(fit_stex_all8, keep = "z_")$stat),
  p = unname(wald(fit_stex_all8, keep = "z_")$p)
)

stex_block_tests <- rbindlist(list(stex_wald_mech, stex_wald_all8))
fwrite(
  stex_block_tests,
  file.path(out_dir, "tables", "05d_stex_speaking_block_tests.csv")
)
save(
  stex_block_tests,
  file = file.path(out_dir, "rdata", "05d_stex_speaking_block_tests.RData"),
  compress = "xz"
)

stex_pair_inf <- stex_inf
fwrite(stex_pair_inf, file.path(out_dir, "tables", "06_stex_pair_sensitivity_results.csv"))
save(stex_pair_inf, file = file.path(out_dir, "rdata", "06_stex_pair_sensitivity_results.RData"), compress = "xz")

save_objects(
  c("toefl_inf", "stex_inf", "stex_pair_inf", "p_toefl_coef", "p_stex_coef"),
  "02_inferential_artifacts.RData"
)

# predictive analyses for TOEFL

toefl_specs <- list(
  list(name = "baseline_mean", type = "baseline_mean", rhs = NULL),
  list(name = "ridge_mechanistic", type = "glmnet", rhs = paste(z_mechanistic_vars, collapse = " + ")),
  list(name = "ridge_all8", type = "glmnet", rhs = paste(z_distance_vars, collapse = " + "))
)

toefl_pred_list <- list()

for (y in toefl_outcomes) {
  dat_y <- copy(toefl[complete.cases(toefl[, c(y, z_distance_vars), with = FALSE])])
  folds <- make_row_vfolds(nrow(dat_y), v = 5, repeats = 1, seed = 1000)

  res <- evaluate_glmnet_cv(
    data = dat_y,
    outcome = y,
    specs = toefl_specs,
    folds = folds,
    family = "gaussian"
  )

  res[, `:=`(
    dataset = "toefl",
    outcome = y,
    scheme = "5fold"
  )]

  toefl_pred_list[[length(toefl_pred_list) + 1L]] <- res
  gc()
}

toefl_pred_fold <- rbindlist(toefl_pred_list, use.names = TRUE, fill = TRUE)
toefl_pred_summary <- toefl_pred_fold[, .(
  mean_rmse = mean(rmse, na.rm = TRUE),
  sd_rmse   = sd(rmse, na.rm = TRUE),
  mean_mae  = mean(mae, na.rm = TRUE),
  sd_mae    = sd(mae, na.rm = TRUE),
  mean_oos_r2 = mean(oos_r2, na.rm = TRUE),
  sd_oos_r2   = sd(oos_r2, na.rm = TRUE)
), by = .(dataset, outcome, scheme, model_name)]

fwrite(toefl_pred_fold,    file.path(out_dir, "tables", "07_toefl_predictive_fold_metrics.csv"))
fwrite(toefl_pred_summary, file.path(out_dir, "tables", "08_toefl_predictive_summary.csv"))
save(toefl_pred_fold, toefl_pred_summary, file = file.path(out_dir, "rdata", "08_toefl_predictive_artifacts.RData"), compress = "xz")

p_toefl_perf <- perf_bar_plot(
  toefl_pred_summary,
  metric = "mean_rmse",
  title_text = "TOEFL predictive performance (5-fold CV; lower RMSE is better)"
)
save_plot(p_toefl_perf, "05_toefl_predictive_rmse.png", width = 14, height = 8)

# predictive analyses for STEX

stex_baseline_rhs <- "AaA + LoR + Edu.day + Sex + Enroll + C + L2 + Family"

stex_specs <- list(
  list(name = "baseline_mean", type = "baseline_mean", rhs = NULL),
  list(name = "ridge_baseline", type = "glmnet", rhs = stex_baseline_rhs),
  list(
    name = "ridge_baseline_plus_mechanistic",
    type = "glmnet",
    rhs = paste(stex_baseline_rhs, paste(z_mechanistic_vars, collapse = " + "), sep = " + ")
  ),
  list(
    name = "ridge_baseline_plus_all8",
    type = "glmnet",
    rhs = paste(stex_baseline_rhs, paste(z_distance_vars, collapse = " + "), sep = " + ")
  )
)

run_stex_predictive <- function(dat, outcome, family = c("gaussian", "poisson")) {
  family <- match.arg(family)

  union_rhs <- paste(
    stex_baseline_rhs,
    paste(z_distance_vars, collapse = " + "),
    sep = " + "
  )

  vars_needed <- unique(all.vars(as.formula(paste0(outcome, " ~ ", union_rhs))))
  dat_use <- copy(dat[complete.cases(dat[, ..vars_needed])])

  folds <- make_group_vfolds(dat_use$pair_id, v = 5, repeats = 1, seed = 2024)

  res <- evaluate_glmnet_cv(
    data = dat_use,
    outcome = outcome,
    specs = stex_specs,
    folds = folds,
    family = family
  )

  res[, `:=`(
    dataset = "stex",
    outcome = outcome,
    scheme = "pair_5fold"
  )]

  res
}

stex_pred_list <- list()

stex_pred_list[[length(stex_pred_list) + 1L]] <- run_stex_predictive(
  dat = stex_main,
  outcome = "Speaking",
  family = "gaussian"
)

for (y in c("new_feat", "new_sounds")) {
  dat_y <- copy(stex_main[!is.na(get(y))])
  stex_pred_list[[length(stex_pred_list) + 1L]] <- run_stex_predictive(
    dat = dat_y,
    outcome = y,
    family = "poisson"
  )
  gc()
}

stex_pred_fold <- rbindlist(stex_pred_list, use.names = TRUE, fill = TRUE)

stex_pred_summary <- stex_pred_fold[, .(
  mean_rmse = mean(rmse, na.rm = TRUE),
  sd_rmse   = sd(rmse, na.rm = TRUE),
  mean_mae  = mean(mae, na.rm = TRUE),
  sd_mae    = sd(mae, na.rm = TRUE),
  mean_oos_r2 = mean(oos_r2, na.rm = TRUE),
  sd_oos_r2   = sd(oos_r2, na.rm = TRUE),
  mean_poisson_dev = mean(poisson_dev, na.rm = TRUE),
  sd_poisson_dev   = sd(poisson_dev, na.rm = TRUE)
), by = .(dataset, outcome, scheme, model_name)]

fwrite(stex_pred_fold,    file.path(out_dir, "tables", "09_stex_predictive_fold_metrics.csv"))
fwrite(stex_pred_summary, file.path(out_dir, "tables", "10_stex_predictive_summary.csv"))
save(stex_pred_fold, stex_pred_summary, file = file.path(out_dir, "rdata", "10_stex_predictive_artifacts.RData"), compress = "xz")

p_stex_perf_speaking <- perf_bar_plot(
  stex_pred_summary[outcome == "Speaking"],
  metric = "mean_rmse",
  title_text = "STEX Speaking predictive performance (pair-grouped 5-fold CV)"
)
save_plot(p_stex_perf_speaking, "06_stex_speaking_predictive_rmse.png", width = 14, height = 8)

p_stex_perf_counts <- perf_bar_plot(
  stex_pred_summary[outcome %in% c("new_feat", "new_sounds")],
  metric = "mean_poisson_dev",
  title_text = "STEX count predictive performance (pair-grouped 5-fold CV)"
)
save_plot(p_stex_perf_counts, "07_stex_counts_predictive_deviance.png", width = 14, height = 8)

save_objects(
  c(
    "toefl_pred_fold",
    "toefl_pred_summary",
    "stex_pred_fold",
    "stex_pred_summary",
    "p_toefl_perf",
    "p_stex_perf_speaking",
    "p_stex_perf_counts"
  ),
  "04_predictive_artifacts.RData"
)

# summary tables

toefl_best_single <- toefl_inf[
  model_block == "single_distance" & term != "(Intercept)"
][order(outcome, p.value)][
  , .SD[1], by = outcome
]

fwrite(toefl_best_single, file.path(out_dir, "tables", "11_toefl_best_single_distance_terms.csv"))

stex_best_single <- stex_inf[
  outcome == "Speaking" &
    model_block == "single_distance" &
    term != "(Intercept)"
][order(outcome, p.value)][
  , .SD[1], by = outcome
]

fwrite(stex_best_single, file.path(out_dir, "tables", "12_stex_best_single_distance_terms.csv"))

toefl_best_pred <- toefl_pred_summary[order(outcome, mean_rmse)][, .SD[1], by = .(outcome, scheme)]

stex_best_pred <- stex_pred_summary[
  order(outcome, scheme, fifelse(outcome == "Speaking", mean_rmse, mean_poisson_dev))
][
  ,
  .SD[1],
  by = .(outcome, scheme)
]

fwrite(toefl_best_pred, file.path(out_dir, "tables", "13_toefl_best_predictive_models.csv"))
fwrite(stex_best_pred,  file.path(out_dir, "tables", "14_stex_best_predictive_models.csv"))

save_objects(
  c("toefl_best_single", "stex_best_single", "toefl_best_pred", "stex_best_pred"),
  "05_summary_tables.RData"
)

# final save and console summary

save.image(file = file.path(out_dir, "rdata", "99_analysis_workspace.RData"), compress = "xz")

cat("\nanalysis complete.\n")
cat("outputs written to: ", normalizePath(out_dir), "\n\n")

cat("filter configuration:\n")
cat("  - OUTLIER_FILTER_MODE: ", outlier_filter_mode, "\n", sep = "")
cat("  - OUTLIER_FILTER_SCOPE: ", outlier_filter_scope, "\n", sep = "")
cat("  - OUTLIER_Z_THRESHOLD: ", outlier_z_threshold, "\n", sep = "")
cat("  - OUTLIER_LEVERAGE_MULT: ", outlier_leverage_mult, "\n", sep = "")
cat("  - OUTLIER_USE_SELECTED_ONLY: ", outlier_use_selected_only, "\n", sep = "")
cat("  - TOEFL_OUTLIER_INCLUDE_OUTCOME: ", toefl_outlier_include_outcome, "\n", sep = "")
cat("  - STEX_OUTLIER_INCLUDE_OUTCOME: ", stex_outlier_include_outcome, "\n\n", sep = "")

cat("stex plotting / scaling configuration:\n")
cat("  - STEX_DISTANCE_STANDARDIZATION_LEVEL: ", stex_distance_standardization_level, "\n", sep = "")
cat("  - STEX_SCATTER_WEIGHTED_SMOOTH: ", stex_scatter_weighted_smooth, "\n\n", sep = "")

cat("key files:\n")
cat("  - 00_filter_summary.csv\n")
cat("  - 00_toefl_filter_log.csv\n")
cat("  - 00_stex_pair_filter_log.csv\n")
cat("  - 00a_toefl_dropped_rows.csv\n")
cat("  - 00b_stex_pair_dropped_rows.csv\n")
cat("  - 00c_stex_main_rows_from_dropped_pairs.csv\n")
cat("  - 00d_toefl_drop_reason_summary.csv\n")
cat("  - 00e_stex_pair_drop_reason_summary.csv\n")
cat("  - 04_toefl_inferential_results.csv\n")
cat("  - 05_stex_inferential_results.csv\n")
cat("  - 08_toefl_predictive_summary.csv\n")
cat("  - 10_stex_predictive_summary.csv\n")
cat("  - 11_toefl_best_single_distance_terms.csv\n")
cat("  - 12_stex_best_single_distance_terms.csv\n")
cat("  - 13_toefl_best_predictive_models.csv\n")
cat("  - 14_stex_best_predictive_models.csv\n")
cat("  - rdata/00_filter_artifacts.RData\n")
cat("  - rdata/00b_dropped_row_artifacts.RData\n")
cat("  - rdata/99_analysis_workspace.RData\n")